# MLE Challenge — Flight Delay Prediction on GCP

## Local environment

```bash
brew install uv libomp   # libomp: OpenMP runtime XGBoost needs on macOS
make install
make model-test
```

---

## Part I — Model implementation

### Result

`make model-test` → **40/40 tests passing**. The target covers more than `model.py`: it also
exercises `config.py`, the connectors (`local`, `gcs`, `bigquery`) and `GCPModelMixin`.

### Bugs found in `exploration.ipynb`

None of these affect the final model (which only uses `OPERA`, `TIPOVUELO` and `MES`), but they
do affect the exploratory analysis the DS relied on.

| # | Where | Bug | Measured impact |
|---|---|---|---|
| 1 | `get_period_day` | Strict comparisons (`>` / `<`) leave out the edges of each range (05:00, 11:59:xx, 12:00, 18:59:xx, 19:00, 23:59:xx, 00:00, 04:59:xx), so the function returns `None`. | **1,230 flights** with no period of day |
| 2 | `is_high_season` | Each range's upper bound (`31-Dec`, `3-Mar`, `31-Jul`, `30-Sep`) is built at 00:00, so any flight on that day after midnight falls outside it. | **730 flights** (100% of flights on those days) labelled low season |
| 3 | `get_rate_from_column` | Computes `total / delays` instead of `delays / total`, and labels it "Tasa (%)" (rate). | Grupo LATAM: actual rate **17.9%**, notebook shows **5.6** |
| 4 | Plots | `sns.barplot(x, y)` with positional arguments fails on `seaborn>=0.12`, the version pinned in `pyproject.toml`. | Plot cells don't run |
| 5 | Cell 37 | `training_data = shuffle(...)` is computed and never used; features are built from `data`. | Dead code (harmless: `train_test_split` already shuffles) |
| 6 | Cell 47 | Thresholds `predict()` output with `> 0.5`, but `predict()` already returns 0/1. | Redundant |

In the implementation, `delay` is computed vectorised (`pd.to_datetime` with an explicit
format) instead of the notebook's row-by-row `apply`. Same rule (`min_diff > 15`), much faster.

### Model selection: XGBoost

XGBoost on the top-10 features, with class balancing (`scale_pos_weight = n_negative / n_positive`).

| Model | Class-1 recall | Class-1 F1 | CV class-1 recall | ROC-AUC |
|---|---|---|---|---|
| **XGBoost, balanced** | 0.688 | 0.366 | 0.689 ± 0.006 | 0.643 |
| LogisticRegression, balanced | 0.688 | 0.364 | 0.647 ± 0.038 | 0.640 |
| XGBoost, unbalanced | 0.006 | 0.012 | 0.006 ± 0.002 | 0.642 |
| LogisticRegression, unbalanced | 0.013 | 0.025 | 0.013 ± 0.003 | 0.640 |

- **Balancing matters more than the algorithm.** Only 18.5% of flights are delayed; without
  re-weighting, both models catch under 2% of delays.
- **Balanced, they tie on quality:** F1 0.366 vs 0.364, with 93% of predictions identical.
- **XGBoost wins on stability:** its class-1 recall varies 6× less across 5-fold CV, keeping a
  safer margin over the tests' 0.60 minimum. It also handles feature interactions natively, and
  the provided interface already expects it.

**Trade-off:** less interpretable than logistic regression's signed coefficients.
**Limitation:** ROC-AUC of 0.64. Real gains would come from better features, not a different
algorithm.

Reproduce with `uv run python docs/model_comparison.py`.

### Design decisions

- **`preprocess` uses `reindex(columns=FEATURES_COLS, fill_value=0)`.** A small serving batch
  may not contain every airline or month; without this, `get_dummies` would produce different
  columns than training. `MES` is cast to `int` so a value read as `7.0` yields `MES_7`, not
  `MES_7.0`.
- **`preprocess` validates its input columns** and fails with a clear message if `OPERA`,
  `TIPOVUELO` or `MES` is missing. If `delay` is already in the data it is used as-is; otherwise
  it is derived from `Fecha-I`/`Fecha-O`.
- **`predict` on an untrained model returns 0 for every row and logs a warning.** The test
  contract requires it (`test_model_predict` and the serving test call `predict` without `fit`).
  Production never takes this path: the serving pipeline loads an artifact explicitly and fails
  loudly if none exists.
- **`fit` rejects single-class data**, where `scale_pos_weight` is undefined.
- **`GCPModelMixin` uses dependency injection** (`reader`, `store`). The same `DelayModel` runs
  locally (`LocalCSVClient` + `LocalArtifactStore`) or on GCP (`BigQueryClient` /
  `GCSCSVClient` + `GCSArtifactStore`) without code changes.
- **Artifacts are versioned by `run_id` with a `latest.txt` pointer**, both locally and on GCS.
  The pointer is written *after* the model, so it never points at a missing artifact.
- **`build_predictions_frame`** (in `connectors/protocols.py`) centralises the output layout
  (`OPERA, TIPOVUELO, MES, prediction`) so every sink writes exactly the same thing, and checks
  that the number of predictions matches the number of rows.
- **`Settings` validates `gcp` mode**, requiring every GCP variable and reporting *all* missing
  ones in a single error rather than one at a time.
- **The BigQuery predictions table has an explicit schema** (`MES INTEGER`,
  `prediction INTEGER`) instead of relying on type inference, and is written with
  `WRITE_TRUNCATE`.

### Notes on the provided repository

- `tests/pipelines/test_training.py` uses `FEATURES_COLS` without importing it (`test_serving.py`
  does import it), so its tests fail with `NameError` regardless of the implementation.
  Fixed in Part III by adding the missing import, following the brief's "If you find any bug,
  fix it". No assertion was changed. The "cannot change the unit test implementation" rule
  applies to `make model-test`, whose tests are untouched.
- In the `Makefile`, `.PHONY: pipeline-test` doesn't match the actual target, `pipelines-test`.
- There is no `.coveragerc`, although the test targets reference it; `coverage` falls back to
  its defaults.
- The repository shipped with 29 `__pycache__` files under version control, compiled on another
  machine, and no `.gitignore`. I added a `.gitignore` and confirmed the tests also pass when
  compiled from source (`PYTHONPYCACHEPREFIX` pointing at a clean directory).

---

## Part II — Data infrastructure

Everything is provisioned by [`infra/bootstrap.sh`](../infra/bootstrap.sh) with `gcloud` and
`bq`; nothing is created from the web console.

```bash
gcloud auth login
GCP_PROJECT_ID=<project-id> ./infra/bootstrap.sh
DRY_RUN=1 GCP_PROJECT_ID=<project-id> ./infra/bootstrap.sh   # print the commands only
```

### Resources

| Resource | Name (defaults) | Purpose |
|---|---|---|
| Service account | `mle-challenge@<project>.iam.gserviceaccount.com` | Single identity for every workload |
| Bucket | `gs://<project>-mle-input` | Raw inputs: `data.csv`, `serving_input.csv` |
| Bucket | `gs://<project>-mle-artifacts` | Versioned models: `<run_id>/model.joblib` + `latest.txt` |
| BigQuery dataset | `mle_challenge` | All tables, in the same region as the buckets |
| Table | `raw_flights` | Training data, loaded from `data.csv` (68,206 rows) |
| Table | `serving_input` | Serving data, loaded from `serving_input.csv` (100 rows) |
| Table | `predictions` | Written by the serving pipeline (Part IV) |

### Design decisions

- **Idempotent.** Each resource is created only if missing and the loads use `--replace`, so
  the script can be re-run safely, for example from CI.
- **Variable names match `challenge/config.py`** (`GCP_PROJECT_ID`, `GCP_BUCKET_INPUT`, …). The
  script prints the resulting runtime configuration, ready to pass to the jobs and the API.
- **`DRY_RUN=1`** prints every command without touching GCP, to review the plan first.
- **Least privilege for the single service account.** Storage access is granted per bucket:
  read-only on inputs, read/write on artifacts. BigQuery needs `jobUser` and `dataEditor`, granted
  at project level.
- **Explicit schemas instead of autodetect.** Flight numbers mix digits and letters (`Vlo-I`),
  so they are `STRING`; dates are `DATETIME`. BigQuery column names cannot contain `-` or `Ñ`,
  so the raw table uses `Fecha_I`, `ANIO`, and so on; `BigQueryClient` restores the source names
  on read.
- **`assets/load_data.sql` selects only the five columns the model needs**, instead of
  `SELECT *`, which reduces the bytes scanned and billed.
- **Serving reads its CSV from Cloud Storage** (`GCSCSVClient`, as the provided interface and
  tests require). The `serving_input` table is loaded too, as the brief asks, and keeps both
  datasets queryable side by side.

### Verification

Run against project `ml-test-513500`:

- **Tables:** `raw_flights` has 68,206 rows and `serving_input` has 100, matching both CSVs.
- **Reads through the project's own connectors:** `BigQueryClient.read()` returns the five
  expected columns with their source names restored. `GCSCSVClient.read()` returns the serving
  input.
- **Same model inputs as the CSV path:** the features and target built from BigQuery equal those
  built from `data.csv`, compared order-independently since BigQuery doesn't guarantee row order
  (delay rate 18.49% in both).
- **Idempotency:** a second run exits cleanly, skips existing resources and leaves the row
  counts unchanged.
- **Data checks before loading:** both CSVs are valid UTF-8; every date matches the `DATETIME`
  format and every day, month and year is an integer. The only null is one flight number
  (`Vlo-O`), loaded as `NULL`.

---

## Part III — Training pipeline

[`challenge/training.py`](../challenge/training.py) runs as the Cloud Run Job `mle-training`.

```bash
make training-local                                          # data/data.csv -> artifacts/
gcloud run jobs execute mle-training --region=us-central1 --wait   # BigQuery -> GCS
```

### Steps

| Step | What it does |
|---|---|
| `step_load_data` | Reads raw flights through the model's injected reader (`LocalCSVClient` or `BigQueryClient`) |
| `step_preprocess` | `DelayModel.preprocess` → top-10 features and `delay` target |
| `step_train` | Fits on a 67/33 split and evaluates on the holdout, then **refits on all rows** for the shipped model |
| `step_save` | Persists `model.joblib` and `metadata.json` under the run's version, then moves `latest.txt` |

`main()` only picks the I/O implementations from `DEPLOYMENT_MODE` (`local`: CSV + local
folder; `gcp`: BigQuery + GCS). The steps are identical in both modes.

### Design decisions

- **Every run is a new version**, `<UTC timestamp>-<random suffix>` (e.g.
  `20261102T022944Z-e93240`): sortable, unique and never overwritten. `latest.txt` is the
  promotion pointer, so rolling back means rewriting one line.
- **`metadata.json` next to each model** stores the version, timestamp, row counts and holdout
  metrics. The API triggers the job and cannot receive return values from it, so this file is how
  `/pipeline/train` (Part V) reports what was trained.
- **Holdout metrics, then refit on all data.** The metrics describe a model trained on 67% of the
  rows; the shipped model uses 100%, so it is at least as good as reported.
- **Image** (`Dockerfile.training`): `python:3.12-slim` plus `libgomp1` (XGBoost's OpenMP runtime);
  dependencies installed with `uv sync --frozen` from `uv.lock`, in their own layer so code-only
  changes rebuild in seconds; runs as a non-root user. `.dockerignore` keeps data, tests and caches
  out of the image.
- **Built by Cloud Build as the single service account** (`infra/cloudbuild.yaml`): native
  `linux/amd64` images without local Docker, and no use of the default Compute/Cloud Build
  accounts. Images are tagged with the commit; uncommitted builds get a `-dirty-<timestamp>`
  suffix so a tag never claims code it doesn't contain.
- **Job settings:** `--max-retries=0` (training is deterministic, so a retry would fail the same
  way and only add cost), `--task-timeout=15m`, 1 CPU / 2 GiB.
- **Provisioning from scratch:** right after `bootstrap.sh` enables the Cloud Build API, the
  first build can be rejected while the API propagates, so builds are retried up to 4 times.

### Verification

- **Unit tests:** `tests/pipelines/test_training.py` passes 3/3 once its missing
  `FEATURES_COLS` import is added (see "Notes on the provided repository" in Part I).
- **Local:** `make training-local` writes `artifacts/<version>/{model.joblib,metadata.json}` and
  `latest.txt`; holdout F1 (delayed class) 0.366, matching the model comparison.
- **`gcp` mode from a laptop:** same pipeline against the real BigQuery table and GCS bucket,
  before containerizing, to separate code issues from container issues.
- **Cloud Run:** execution `mle-training-nh967` succeeded and published version
  `20261102T022944Z-e93240` with the same metrics.
- **`bootstrap.sh` end to end** (exit code 0), including the registry, the build and the job.

### Known limitation

BigQuery doesn't guarantee row order, so the holdout split may differ between runs on the same
data. Metrics can vary slightly from run to run; the shipped model is unaffected because it is
refit on all rows.

---

## Part IV — Serving pipeline

[`challenge/serving.py`](../challenge/serving.py) runs as the Cloud Run Job `mle-serving`.

```bash
make serving-local                                          # latest local model -> artifacts/predictions.csv
gcloud run jobs execute mle-serving --region=us-central1 --wait   # GCS input -> BigQuery predictions
```

### Steps

| Step | What it does |
|---|---|
| `step_load_model` | Loads the model `latest.txt` points to (or a given version) |
| `step_load_data` | Reads `serving_input.csv` (`LocalCSVClient` or `GCSCSVClient`) |
| `step_preprocess` | `DelayModel.preprocess` without a target → top-10 features |
| `step_predict` | `DelayModel.predict` → one 0/1 prediction per flight |
| `step_write` | Writes `OPERA, TIPOVUELO, MES, prediction` to `artifacts/predictions.csv` or the BigQuery `predictions` table |

### Design decisions

- **Fails loudly without a model.** `step_load_model` raises when no artifact exists, so the job
  exits with code 1 and the message "train a model first". It never reaches
  `DelayModel.predict`'s untrained fallback, which returns all zeros.
- **Predictions are written through the model** (`GCPModelMixin.write_predictions`, a new public
  method), so the pipeline never reaches into the model's private reader.
- **The `predictions` table is replaced on each run** (`WRITE_TRUNCATE`) with an explicit schema:
  it always holds the output of the latest batch, which is what the API reads.
- **Same image recipe as training**: `Dockerfile.serving` differs only in its command, and
  `bootstrap.sh` builds and deploys both jobs with the same functions and job settings.

### Verification

- **Unit tests:** `make pipelines-test` passes 8/8 (3 training + 5 serving).
- **Local:** `make serving-local` scores the 100 flights of `serving_input.csv` (52 predicted
  delays).
- **No model:** with an empty artifacts directory the job exits with code 1 and "train a model
  first".
- **`gcp` mode from a laptop:** the `predictions` table receives 100 rows, with the expected schema
  and the same 52 delays as the local run.
- **Cloud Run:** execution `mle-serving-pxvt6` succeeded and rewrote the `predictions` table
  (100 rows, 52 delays).
- **`bootstrap.sh` end to end** (exit code 0): builds both images and deploys both jobs.

## Part V — Pipeline control API

[`challenge/api/api.py`](../challenge/api/api.py) runs as the Cloud Run service `mle-api`.

```bash
make run-api-local    # local mode: pipelines run in-process, no GCP credentials needed
make api-test
make stress-test      # against STRESS_URL (default http://127.0.0.1:8000)
```

### Endpoints

| Endpoint | Behavior | Errors |
|---|---|---|
| `GET /health` | `{"status": "ok"}` | — |
| `POST /pipeline/train` | Runs `mle-training` and waits; returns the execution id plus the new model's `metadata.json` (version, row counts, metrics) | 422 if `raw_flights` is empty, 409 if a training run is in progress, 500 if the job fails |
| `POST /pipeline/predict` | Runs `mle-serving` and waits; returns the execution id, the total and the first 10 predictions | 422 if there is no trained model, 409 if a serving run is in progress, 500 if the job fails |
| `GET /pipeline/predict/results` | Paginated predictions; `opera`, `tipovuelo`, `mes` take comma-separated lists | 422 for an invalid `mes`, `page < 1` or `page_size` outside 1–100 |

### Design decisions

- **Synchronous runs through the Cloud Run Jobs API** (`run_job(...).result()`), as the README
  asks. The service timeout is 15 minutes, the same as the jobs' task timeout.
- **One run per pipeline at a time.** A second `POST` while a run is in progress returns 409,
  so a public endpoint cannot stack parallel executions, and `/train` always returns the metadata
  of its own run. The check combines a lock per instance with Cloud Run's list of active
  executions, so it holds across instances and for runs started by hand.
- **Sync endpoints.** FastAPI runs them in its threadpool, so a training run that blocks for
  minutes does not stall `/health` or `/results` on the same instance.
- **Results are served from memory.** Each instance keeps a snapshot of the `predictions` table
  for 30 seconds (refreshed right after `/pipeline/predict`) instead of querying BigQuery on every
  request. The table only changes when the serving job runs, and the stress test then measures
  the API rather than BigQuery's per-query latency and cost.
- **Stable ordering.** The `predictions` table has no row id, so pages are sorted by
  `OPERA, TIPOVUELO, MES`. "First 10 predictions" means the first page in that order.
- **Local mode** runs the same pipelines in-process (`training.main`, `serving.main`), so the API
  and the stress test work without GCP.
- **Least privilege:** the service account can run only the two jobs (`roles/run.invoker` on each).
  Waiting on a run needs `run.operations.get`, which exists only at project level, so it also gets
  the read-only `roles/run.viewer` on the project. `--max-instances=3` caps the cost of a public endpoint.
- **Pre-flight checks are free:** the row count comes from table metadata, not a query.

### Verification

- **Unit tests:** `make api-test` passes 16/16 (the 14 provided plus two for the 409); `make model-test` (40) and `make pipelines-test`
  (8) still pass.
- **Local, end to end:** train → predict → filtered results all return the expected payloads;
  invalid `mes` and `page_size` return 422; a second `/train` during a run returns 409.
- **Stress test (local):** 100 users for 60 s, ~168,000 requests, 0 failures, median 10 ms.
- **Cloud Run** (`https://mle-api-475124136438.us-central1.run.app`): `/pipeline/train` returned the
  new model's metadata after 113 s (execution `mle-training-c2jrk`), `/pipeline/predict` ran
  `mle-serving-zvr6w` and returned the first 10 of 100 predictions, filtered results match the
  local run, and a `/train` during a run returned 409.
- **Stress test (deployed):** 100 users for 60 s from a laptop, 9,263 requests, 0 failures, median
  190 ms (mostly network latency).
- **`bootstrap.sh` end to end** (exit code 0): builds the three images, deploys both jobs and the API.
