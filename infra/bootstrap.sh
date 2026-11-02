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
DRY_RUN="${DRY_RUN:-0}"

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
  bigquery.googleapis.com

# --- 2. service account (the single identity for every workload) -------------

log "Service account ${SERVICE_ACCOUNT_EMAIL}"
if ! exists gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}"; then
  run gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name="MLE challenge pipelines and API"
fi
# Project-level BigQuery roles: run query/load jobs and read/write tables.
for role in roles/bigquery.jobUser roles/bigquery.dataEditor; do
  run gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet --format=none
done

# --- 3. Cloud Storage --------------------------------------------------------

for bucket in "${GCP_BUCKET_INPUT}" "${GCP_BUCKET_ARTIFACTS}"; do
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
