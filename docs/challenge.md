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
  Addressed in Part III.
- In the `Makefile`, `.PHONY: pipeline-test` doesn't match the actual target, `pipelines-test`.
- There is no `.coveragerc`, although the test targets reference it; `coverage` falls back to
  its defaults.
- The repository shipped with 29 `__pycache__` files under version control, compiled on another
  machine, and no `.gitignore`. I added a `.gitignore` and confirmed the tests also pass when
  compiled from source (`PYTHONPYCACHEPREFIX` pointing at a clean directory).
