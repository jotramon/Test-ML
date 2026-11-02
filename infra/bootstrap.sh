#!/usr/bin/env bash
# Provision the challenge's GCP infrastructure from scratch with gcloud/bq.
#
# Idempotent: every resource is created only if it does not exist yet, and the
# BigQuery loads replace their tables, so the script is safe to re-run.
#
# Usage:
#   GCP_PROJECT_ID=my-project ./infra/bootstrap.sh
#   DRY_RUN=1 GCP_PROJECT_ID=my-project ./infra/bootstrap.sh   # print commands only
#   DEPLOY_ONLY=1 GCP_PROJECT_ID=my-project ./infra/bootstrap.sh # build + deploy (CD)
#
# A full run provisions everything (APIs, IAM, buckets, BigQuery, registry,
# GitHub Actions identity) and then builds and deploys. DEPLOY_ONLY=1 skips the
# provisioning, which needs project-owner rights, so CD can run as the
# challenge's service account.
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
BQ_PREDICTIONS_TABLE="${BQ_PREDICTIONS_TABLE:-predictions}"
TRAINING_JOB="${TRAINING_JOB:-mle-training}"
SERVING_JOB="${SERVING_JOB:-mle-serving}"
API_SERVICE="${API_SERVICE:-mle-api}"
WIF_POOL="${WIF_POOL:-github}"
WIF_PROVIDER="${WIF_PROVIDER:-github-actions}"
# owner/repo allowed to deploy through Workload Identity Federation; defaults
# to the origin remote, and stays empty (federation skipped) outside a clone.
github_repository_from_remote() {
  local url
  url="$(git -C "$(dirname "${BASH_SOURCE[0]}")" remote get-url origin 2>/dev/null)" || return 0
  sed -nE 's#.*github\.com[:/]([^/]+/[^/.]+)(\.git)?$#\1#p' <<< "${url}"
}
GITHUB_REPOSITORY="${GITHUB_REPOSITORY:-$(github_repository_from_remote)}"
DEPLOY_ONLY="${DEPLOY_ONLY:-0}"
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

# True unless DEPLOY_ONLY=1.
provisioning() { [[ "${DEPLOY_ONLY}" != "1" ]]; }

# Succeed if the resource checked by the given command exists.
# In dry-run mode nothing is queried and every resource counts as missing.
exists() {
  [[ "${DRY_RUN}" == "1" ]] && return 1
  "$@" >/dev/null 2>&1
}

log "Project ${GCP_PROJECT_ID} (${GCP_REGION})"
run gcloud config set project "${GCP_PROJECT_ID}"

if provisioning; then

# --- 1. APIs -----------------------------------------------------------------

run gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
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
# Project-level roles:
#   runtime  BigQuery query/load jobs and table access; logs (also needed by
#            Cloud Build running as this account); run.viewer so the API can
#            wait on job runs (run.operations.get exists only at project level).
#   deploy   (CD runs as this account) submitting builds, reading their logs,
#            calling APIs in the project.
PROJECT_ROLES=(
  roles/bigquery.jobUser roles/bigquery.dataEditor roles/logging.logWriter roles/run.viewer
  roles/cloudbuild.builds.editor roles/logging.viewer roles/serviceusage.serviceUsageConsumer
)
for role in "${PROJECT_ROLES[@]}"; do
  run gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet --format=none
done
# Deploying jobs/services and running builds *as* this account requires actAs
# on it, granted on the account itself rather than project-wide.
run gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role=roles/iam.serviceAccountUser --format=none

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
# The build bucket only holds source tarballs: CD uploads them (gcloud also
# checks the bucket itself, hence admin on this bucket) and Cloud Build reads them.
run gcloud storage buckets add-iam-policy-binding "gs://${GCP_BUCKET_BUILD}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role=roles/storage.admin --format=none

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

# --- 5. container registry --------------------------------------------------

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
# Every build pushes ~1 GB per image: keep the 5 newest versions of each and
# delete older ones after a week (Keep wins over Delete).
CLEANUP_POLICY="$(mktemp)"
trap 'rm -f "${CLEANUP_POLICY}"' EXIT
cat > "${CLEANUP_POLICY}" <<'POLICY'
[
  {"name": "keep-recent", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 5}},
  {"name": "delete-old", "action": {"type": "Delete"}, "condition": {"olderThan": "7d"}}
]
POLICY
run gcloud artifacts repositories set-cleanup-policies "${AR_REPOSITORY}" \
  --location="${GCP_REGION}" \
  --policy="${CLEANUP_POLICY}" \
  --no-dry-run --format=none

# --- 6. GitHub Actions identity (Workload Identity Federation) ---------------

# CD authenticates with short-lived OIDC tokens instead of a key file. Only
# workflows of GITHUB_REPOSITORY running on main may impersonate the account.
if [[ -n "${GITHUB_REPOSITORY}" ]]; then
  log "Workload Identity Federation for ${GITHUB_REPOSITORY}"
  PROJECT_NUMBER="$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')"
  if ! exists gcloud iam workload-identity-pools describe "${WIF_POOL}" --location=global; then
    run gcloud iam workload-identity-pools create "${WIF_POOL}" \
      --location=global --display-name="GitHub Actions"
  fi
  if ! exists gcloud iam workload-identity-pools providers describe "${WIF_PROVIDER}" \
      --location=global --workload-identity-pool="${WIF_POOL}"; then
    run gcloud iam workload-identity-pools providers create-oidc "${WIF_PROVIDER}" \
      --location=global \
      --workload-identity-pool="${WIF_POOL}" \
      --issuer-uri="https://token.actions.githubusercontent.com" \
      --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
      --attribute-condition="assertion.repository == '${GITHUB_REPOSITORY}' && assertion.ref == 'refs/heads/main'"
  fi
  run gcloud iam service-accounts add-iam-policy-binding "${SERVICE_ACCOUNT_EMAIL}" \
    --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_REPOSITORY}" \
    --role=roles/iam.workloadIdentityUser --format=none
else
  log "No GitHub remote found; skipping Workload Identity Federation"
fi

fi  # provisioning

# --- 7. images ---------------------------------------------------------------

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
SERVING_IMAGE="${REGISTRY}/serving:${IMAGE_TAG}"
API_IMAGE="${REGISTRY}/api:${IMAGE_TAG}"
build_image Dockerfile.training "${TRAINING_IMAGE}"
build_image Dockerfile.serving "${SERVING_IMAGE}"
build_image Dockerfile.api "${API_IMAGE}"

# --- 8. pipelines as Cloud Run Jobs ------------------------------------------

RUNTIME_ENV="DEPLOYMENT_MODE=gcp,GCP_PROJECT_ID=${GCP_PROJECT_ID},GCP_REGION=${GCP_REGION}"
RUNTIME_ENV+=",GCP_BUCKET_INPUT=${GCP_BUCKET_INPUT},GCP_BUCKET_ARTIFACTS=${GCP_BUCKET_ARTIFACTS}"
RUNTIME_ENV+=",BQ_DATASET=${BQ_DATASET},BQ_RAW_TABLE=${BQ_RAW_TABLE}"
RUNTIME_ENV+=",BQ_PREDICTIONS_TABLE=${BQ_PREDICTIONS_TABLE}"

# `deploy` creates the job or updates it in place. No retries: both pipelines
# are deterministic, so a retry would fail the same way and only add cost.
deploy_job() {
  local job="$1" image="$2"
  log "Cloud Run Job ${job}"
  run gcloud run jobs deploy "${job}" \
    --image="${image}" \
    --region="${GCP_REGION}" \
    --service-account="${SERVICE_ACCOUNT_EMAIL}" \
    --set-env-vars="${RUNTIME_ENV}" \
    --tasks=1 \
    --max-retries=0 \
    --task-timeout=15m \
    --cpu=1 \
    --memory=2Gi
}

deploy_job "${TRAINING_JOB}" "${TRAINING_IMAGE}"
deploy_job "${SERVING_JOB}" "${SERVING_IMAGE}"

# --- 9. control API as a Cloud Run service -----------------------------------

# --timeout covers a synchronous training run; --max-instances caps the cost of
# a public endpoint. Public access is granted below, as an IAM binding, so a
# DEPLOY_ONLY run never needs permission to change it.
log "Cloud Run service ${API_SERVICE}"
run gcloud run deploy "${API_SERVICE}" \
  --image="${API_IMAGE}" \
  --region="${GCP_REGION}" \
  --service-account="${SERVICE_ACCOUNT_EMAIL}" \
  --set-env-vars="${RUNTIME_ENV},TRAINING_JOB=${TRAINING_JOB},SERVING_JOB=${SERVING_JOB}" \
  --timeout=900 \
  --cpu=1 \
  --memory=1Gi \
  --max-instances=3 \
  --quiet

# --- 10. access to the deployed resources ------------------------------------

if provisioning; then
  # run.developer, scoped to the two jobs and the API: the API runs the jobs,
  # and CD redeploys all three, without project-wide Cloud Run rights.
  for job in "${TRAINING_JOB}" "${SERVING_JOB}"; do
    run gcloud run jobs add-iam-policy-binding "${job}" \
      --region="${GCP_REGION}" \
      --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
      --role=roles/run.developer --format=none
  done
  run gcloud run services add-iam-policy-binding "${API_SERVICE}" \
    --region="${GCP_REGION}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role=roles/run.developer --format=none
  # Public, because evaluators call the API directly.
  run gcloud run services add-iam-policy-binding "${API_SERVICE}" \
    --region="${GCP_REGION}" \
    --member=allUsers \
    --role=roles/run.invoker --format=none
fi

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
BQ_PREDICTIONS_TABLE=${BQ_PREDICTIONS_TABLE}
SETTINGS
printf '\nRun the pipelines with:\n'
for job in "${TRAINING_JOB}" "${SERVING_JOB}"; do
  printf '  gcloud run jobs execute %s --region=%s --wait\n' "${job}" "${GCP_REGION}"
done
if provisioning && [[ -n "${GITHUB_REPOSITORY}" && "${DRY_RUN}" != "1" ]]; then
  printf '\nCD (.github/workflows/cd.yml):\n  workload_identity_provider: %s\n  service_account: %s\n' \
    "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}" \
    "${SERVICE_ACCOUNT_EMAIL}"
fi
if [[ "${DRY_RUN}" != "1" ]]; then
  printf '\nAPI: %s\n' "$(gcloud run services describe "${API_SERVICE}" \
    --region="${GCP_REGION}" --format='value(status.url)')"
fi
