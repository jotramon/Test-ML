#!/usr/bin/env bash
# Provision the challenge's GCP infrastructure from scratch with gcloud/bq.
#
# Idempotent: every resource is created only if it does not exist yet, and the
# BigQuery loads replace their tables, so the script is safe to re-run.
#
# Usage:
#   GCP_PROJECT_ID=my-project ./infra/bootstrap.sh
#   DRY_RUN=1 GCP_PROJECT_ID=my-project ./infra/bootstrap.sh   # print commands only
#
# Variable names match challenge/config.py, so the same values configure the
# pipelines and the API.
set -euo pipefail

GCP_PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID to the target project}"
GCP_REGION="${GCP_REGION:-us-central1}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-mle-challenge}"
GCP_BUCKET_INPUT="${GCP_BUCKET_INPUT:-${GCP_PROJECT_ID}-mle-input}"
GCP_BUCKET_ARTIFACTS="${GCP_BUCKET_ARTIFACTS:-${GCP_PROJECT_ID}-mle-artifacts}"
BQ_DATASET="${BQ_DATASET:-mle_challenge}"
BQ_RAW_TABLE="${BQ_RAW_TABLE:-raw_flights}"
BQ_SERVING_TABLE="${BQ_SERVING_TABLE:-serving_input}"
GCP_BUCKET_BUILD="${GCP_BUCKET_BUILD:-${GCP_PROJECT_ID}-mle-build}"
AR_REPOSITORY="${AR_REPOSITORY:-mle-challenge}"
TRAINING_JOB="${TRAINING_JOB:-mle-training}"
DRY_RUN="${DRY_RUN:-0}"

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${AR_REPOSITORY}"
# Tag images with the commit they were built from; uncommitted changes get a
# "-dirty-<timestamp>" suffix so a tag never claims code it doesn't contain.
default_image_tag() {
  local tag
  tag="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null)" || tag="build"
  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
    tag+="-dirty-$(date -u +%Y%m%d%H%M%S)"
  fi
  printf '%s' "${tag}"
}
IMAGE_TAG="${IMAGE_TAG:-$(default_image_tag)}"

# BigQuery column names cannot contain "-" or "Ñ", so the raw table renames
# them (Fecha-I -> Fecha_I, AÑO -> ANIO). BigQueryClient restores the source
# names on read. Flight numbers mix digits and letters, hence STRING.
RAW_FLIGHTS_SCHEMA="Fecha_I:DATETIME,Vlo_I:STRING,Ori_I:STRING,Des_I:STRING,Emp_I:STRING,\
Fecha_O:DATETIME,Vlo_O:STRING,Ori_O:STRING,Des_O:STRING,Emp_O:STRING,\
DIA:INTEGER,MES:INTEGER,ANIO:INTEGER,DIANOM:STRING,TIPOVUELO:STRING,OPERA:STRING,\
SIGLAORI:STRING,SIGLADES:STRING"
SERVING_INPUT_SCHEMA="OPERA:STRING,TIPOVUELO:STRING,MES:INTEGER"

# --- helpers -----------------------------------------------------------------

log() { printf '\n==> %s\n' "$*"; }

# Run a command, or only print it when DRY_RUN=1.
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# Retry a command a few times. Newly enabled APIs (e.g. Cloud Build) take a
# moment to propagate and reject the first calls with PERMISSION_DENIED.
retry() {
  local attempt=1 max_attempts=4
  until "$@"; do
    if (( attempt >= max_attempts )); then
      return 1
    fi
    log "Attempt ${attempt} failed; retrying in 30s"
    sleep 30
    attempt=$((attempt + 1))
  done
}

# Succeed if the resource checked by the given command exists.
# In dry-run mode nothing is queried and every resource counts as missing.
exists() {
  [[ "${DRY_RUN}" == "1" ]] && return 1
  "$@" >/dev/null 2>&1
}

# --- 1. project and APIs -----------------------------------------------------

log "Project ${GCP_PROJECT_ID} (${GCP_REGION})"
run gcloud config set project "${GCP_PROJECT_ID}"
run gcloud services enable \
  iam.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  logging.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com

# --- 2. service account (the single identity for every workload) -------------

log "Service account ${SERVICE_ACCOUNT_EMAIL}"
if ! exists gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}"; then
  run gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name="MLE challenge pipelines and API"
fi
# Project-level roles: BigQuery query/load jobs and table access, and writing
# logs (required for Cloud Build when it runs as this account).
for role in roles/bigquery.jobUser roles/bigquery.dataEditor roles/logging.logWriter; do
  run gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet --format=none
done

# --- 3. Cloud Storage --------------------------------------------------------

for bucket in "${GCP_BUCKET_INPUT}" "${GCP_BUCKET_ARTIFACTS}" "${GCP_BUCKET_BUILD}"; do
  log "Bucket gs://${bucket}"
  if ! exists gcloud storage buckets describe "gs://${bucket}"; then
    run gcloud storage buckets create "gs://${bucket}" \
      --location="${GCP_REGION}" \
      --uniform-bucket-level-access \
      --public-access-prevention
  fi
done
# Bucket-scoped access: pipelines only read inputs, but write model artifacts.
run gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET_INPUT}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role=roles/storage.objectViewer --format=none
run gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET_ARTIFACTS}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role=roles/storage.objectAdmin --format=none
# Cloud Build reads the uploaded source from the build bucket.
run gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET_BUILD}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role=roles/storage.objectViewer --format=none

log "Uploading input data to gs://${GCP_BUCKET_INPUT}"
run gcloud storage cp \
  "${REPO_ROOT}/data/data.csv" \
  "${REPO_ROOT}/data/serving_input.csv" \
  "gs://${GCP_BUCKET_INPUT}/"

# --- 4. BigQuery -------------------------------------------------------------

log "Dataset ${GCP_PROJECT_ID}:${BQ_DATASET}"
if ! exists bq --project_id="${GCP_PROJECT_ID}" show --dataset "${GCP_PROJECT_ID}:${BQ_DATASET}"; then
  run bq --project_id="${GCP_PROJECT_ID}" --location="${GCP_REGION}" mk --dataset \
    --description="MLE challenge: raw flights, serving input and predictions" \
    "${GCP_PROJECT_ID}:${BQ_DATASET}"
fi

log "Loading ${BQ_RAW_TABLE} and ${BQ_SERVING_TABLE} from Cloud Storage"
run bq --project_id="${GCP_PROJECT_ID}" --location="${GCP_REGION}" load \
  --replace --source_format=CSV --skip_leading_rows=1 \
  "${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_RAW_TABLE}" \
  "gs://${GCP_BUCKET_INPUT}/data.csv" \
  "${RAW_FLIGHTS_SCHEMA}"
run bq --project_id="${GCP_PROJECT_ID}" --location="${GCP_REGION}" load \
  --replace --source_format=CSV --skip_leading_rows=1 \
  "${GCP_PROJECT_ID}:${BQ_DATASET}.${BQ_SERVING_TABLE}" \
  "gs://${GCP_BUCKET_INPUT}/serving_input.csv" \
  "${SERVING_INPUT_SCHEMA}"

# --- 5. container registry and images ---------------------------------------

log "Artifact Registry ${REGISTRY}"
if ! exists gcloud artifacts repositories describe "${AR_REPOSITORY}" --location="${GCP_REGION}"; then
  run gcloud artifacts repositories create "${AR_REPOSITORY}" \
    --repository-format=docker \
    --location="${GCP_REGION}" \
    --description="MLE challenge pipeline and API images"
fi
run gcloud artifacts repositories add-iam-policy-binding "${AR_REPOSITORY}" \
  --location="${GCP_REGION}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role=roles/artifactregistry.writer --format=none

# Build in Cloud Build (native linux/amd64, no local Docker needed), as the
# challenge's service account.
build_image() {
  local dockerfile="$1" image="$2"
  log "Building ${image}"
  retry run gcloud builds submit "${REPO_ROOT}" \
    --config="${REPO_ROOT}/infra/cloudbuild.yaml" \
    --substitutions="_DOCKERFILE=${dockerfile},_IMAGE=${image}" \
    --service-account="projects/${GCP_PROJECT_ID}/serviceAccounts/${SERVICE_ACCOUNT_EMAIL}" \
    --gcs-source-staging-dir="gs://${GCP_BUCKET_BUILD}/source" \
    --region="${GCP_REGION}"
}

TRAINING_IMAGE="${REGISTRY}/training:${IMAGE_TAG}"
build_image Dockerfile.training "${TRAINING_IMAGE}"

# --- 6. pipelines as Cloud Run Jobs ------------------------------------------

RUNTIME_ENV="DEPLOYMENT_MODE=gcp,GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_REGION=${GCP_REGION}"
RUNTIME_ENV+=",GCP_BUCKET_INPUT=${GCP_BUCKET_INPUT},GCP_BUCKET_ARTIFACTS=${GCP_BUCKET_ARTIFACTS}"
RUNTIME_ENV+=",BQ_DATASET=${BQ_DATASET},BQ_RAW_TABLE=${BQ_RAW_TABLE}"

# `deploy` creates the job or updates it in place.
log "Cloud Run Job ${TRAINING_JOB}"
run gcloud run jobs deploy "${TRAINING_JOB}" \
  --image="${TRAINING_IMAGE}" \
  --region="${GCP_REGION}" \
  --service-account="${SERVICE_ACCOUNT_EMAIL}" \
  --set-env-vars="${RUNTIME_ENV}" \
  --tasks=1 \
  --max-retries=0 \
  --task-timeout=15m \
  --cpu=1 \
  --memory=2Gi

# --- summary -----------------------------------------------------------------

log "Done. Runtime configuration:"
cat <<SETTINGS
DEPLOYMENT_MODE=gcp
GCP_PROJECT_ID=${GCP_PROJECT_ID}
GCP_REGION=${GCP_REGION}
GCP_BUCKET_INPUT=${GCP_BUCKET_INPUT}
GCP_BUCKET_ARTIFACTS=${GCP_BUCKET_ARTIFACTS}
BQ_DATASET=${BQ_DATASET}
BQ_RAW_TABLE=${BQ_RAW_TABLE}
SETTINGS
printf '\nRun training with:\n  gcloud run jobs execute %s --region=%s --wait\n' \
  "${TRAINING_JOB}" "${GCP_REGION}"
