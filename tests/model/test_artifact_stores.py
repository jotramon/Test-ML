"""Unit tests for challenge.connectors.gcs."""

import io
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest

from challenge.config import clear_settings_cache
from challenge.connectors.gcs import GCSArtifactStore, GCSCSVClient
from challenge.connectors.local import LocalArtifactStore
from challenge.connectors.protocols import ArtifactStore, DataReader
from challenge.model import DelayModel


class _FakeEstimator:
    pass


def _fitted_model() -> DelayModel:
    m = DelayModel()
    m._model = _FakeEstimator()
    return m


def test_save_creates_joblib_and_latest(tmp_path, monkeypatch):
    """save() writes model.joblib and creates latest.txt."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    model = _fitted_model()
    path = LocalArtifactStore().save(model, "run-001")
    assert (tmp_path / "run-001" / "model.joblib").exists()
    assert path == str(tmp_path / "run-001" / "model.joblib")
    assert (tmp_path / "latest.txt").read_text() == "run-001"


def test_save_updates_latest_txt(tmp_path, monkeypatch):
    """Second save overwrites latest.txt with new run_id."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    store = LocalArtifactStore()
    model = _fitted_model()
    store.save(model, "run-001")
    store.save(model, "run-002")
    assert (tmp_path / "latest.txt").read_text() == "run-002"


def test_load_by_explicit_run_id(tmp_path, monkeypatch):
    """load() with an explicit run_id restores model._model."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    store = LocalArtifactStore()
    original = _fitted_model()
    store.save(original, "run-abc")

    target = DelayModel()
    store.load(target, "run-abc")
    assert isinstance(target._model, _FakeEstimator)


def test_load_latest_when_run_id_none(tmp_path, monkeypatch):
    """load(run_id=None) resolves the run_id from latest.txt."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    store = LocalArtifactStore()
    original = _fitted_model()
    store.save(original, "run-xyz")

    target = DelayModel()
    store.load(target, None)
    assert isinstance(target._model, _FakeEstimator)


def test_load_raises_when_latest_txt_missing(tmp_path, monkeypatch):
    """FileNotFoundError is raised when latest.txt is absent and run_id is None."""
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    model = DelayModel()
    with pytest.raises(FileNotFoundError, match="latest.txt"):
        LocalArtifactStore().load(model, None)


def test_custom_artifacts_dir_via_env(tmp_path, monkeypatch):
    """ARTIFACTS_DIR env var controls where artifacts are stored."""
    custom = tmp_path / "custom-artifacts"
    monkeypatch.setenv("ARTIFACTS_DIR", str(custom))
    model = _fitted_model()
    LocalArtifactStore().save(model, "run-env")
    assert (custom / "run-env" / "model.joblib").exists()


def test_local_artifact_store_satisfies_protocol():
    """isinstance check confirms structural compatibility."""
    assert isinstance(LocalArtifactStore(), ArtifactStore)


# ---------------------------------------------------------------------------
# GCSArtifactStore tests
# ---------------------------------------------------------------------------


def _make_gcs_client_mock(
    bucket_name: str = "test-bucket",
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_client_cls, mock_bucket) with pre-wired blob factory."""
    mock_bucket = MagicMock()
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_client_cls = MagicMock(return_value=mock_client)
    return mock_client_cls, mock_bucket


@patch(
    "challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="test-bucket"
)
def test_gcs_save_uploads_joblib_blob(mock_bucket_name):
    """save() uploads <run_id>/model.joblib and latest.txt to GCS."""
    mock_client_cls, mock_bucket = _make_gcs_client_mock()

    with patch("google.cloud.storage.Client", mock_client_cls):
        model = _fitted_model()
        store = GCSArtifactStore()
        uri = store.save(model, "run-gcs-001")

    mock_bucket.blob.assert_any_call("run-gcs-001/model.joblib")
    mock_bucket.blob.assert_any_call("latest.txt")
    assert uri == "gs://test-bucket/run-gcs-001/model.joblib"


@patch(
    "challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="test-bucket"
)
def test_gcs_save_writes_latest_txt(mock_bucket_name):
    """save() uploads run_id as the content of latest.txt."""
    mock_client_cls, mock_bucket = _make_gcs_client_mock()
    latest_blob = MagicMock()
    model_blob = MagicMock()

    def _blob_factory(name: str) -> MagicMock:
        return latest_blob if name == "latest.txt" else model_blob

    mock_bucket.blob.side_effect = _blob_factory

    with patch("google.cloud.storage.Client", mock_client_cls):
        store = GCSArtifactStore()
        store.save(_fitted_model(), "run-gcs-002")

    latest_blob.upload_from_string.assert_called_once_with("run-gcs-002")


@patch(
    "challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="test-bucket"
)
def test_gcs_load_by_explicit_run_id(mock_bucket_name):
    """load() with a run_id downloads and deserializes the model."""
    mock_client_cls, mock_bucket = _make_gcs_client_mock()

    fake_estimator = _FakeEstimator()
    buf = io.BytesIO()
    joblib.dump(fake_estimator, buf)
    buf.seek(0)

    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = buf.read()
    mock_bucket.blob.return_value = mock_blob

    with patch("google.cloud.storage.Client", mock_client_cls):
        target = DelayModel()
        GCSArtifactStore().load(target, "run-gcs-abc")

    mock_bucket.blob.assert_called_with("run-gcs-abc/model.joblib")
    assert isinstance(target._model, _FakeEstimator)


@patch(
    "challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="test-bucket"
)
def test_gcs_load_latest_when_run_id_none(mock_bucket_name):
    """load(run_id=None) resolves run_id from latest.txt then downloads model."""
    mock_client_cls, mock_bucket = _make_gcs_client_mock()

    fake_estimator = _FakeEstimator()
    buf = io.BytesIO()
    joblib.dump(fake_estimator, buf)
    buf.seek(0)
    model_bytes = buf.read()

    latest_blob = MagicMock()
    latest_blob.exists.return_value = True
    latest_blob.download_as_text.return_value = "run-latest\n"

    model_blob = MagicMock()
    model_blob.download_as_bytes.return_value = model_bytes

    def _blob_factory(name: str) -> MagicMock:
        return latest_blob if name == "latest.txt" else model_blob

    mock_bucket.blob.side_effect = _blob_factory

    with patch("google.cloud.storage.Client", mock_client_cls):
        target = DelayModel()
        GCSArtifactStore().load(target, None)

    assert isinstance(target._model, _FakeEstimator)


@patch(
    "challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="test-bucket"
)
def test_gcs_load_raises_when_latest_txt_missing(mock_bucket_name):
    """FileNotFoundError is raised when latest.txt is absent and run_id is None."""
    mock_client_cls, mock_bucket = _make_gcs_client_mock()

    missing_blob = MagicMock()
    missing_blob.exists.return_value = False
    mock_bucket.blob.return_value = missing_blob

    with patch("google.cloud.storage.Client", mock_client_cls):  # noqa: SIM117
        with pytest.raises(FileNotFoundError, match="latest.txt"):  # noqa: RUF043
            GCSArtifactStore().load(DelayModel(), None)


def test_gcs_artifact_store_satisfies_protocol():
    """isinstance check confirms GCSArtifactStore satisfies ArtifactStore protocol."""
    assert isinstance(GCSArtifactStore(), ArtifactStore)


@patch(
    "challenge.connectors.gcs.GCSArtifactStore._bucket_name", return_value="test-bucket"
)
def test_gcs_artifact_store_load_raises_when_model_blob_download_fails(mock_bucket_name):
    """load() propagates an exception when the model blob download fails."""
    mock_client_cls, mock_bucket = _make_gcs_client_mock()

    latest_blob = MagicMock()
    latest_blob.exists.return_value = True
    latest_blob.download_as_text.return_value = "run-ok"

    model_blob = MagicMock()
    model_blob.download_as_bytes.side_effect = Exception("blob missing")

    def _blob_factory(name: str) -> MagicMock:
        return latest_blob if name == "latest.txt" else model_blob

    mock_bucket.blob.side_effect = _blob_factory

    with (
        patch("google.cloud.storage.Client", mock_client_cls),
        pytest.raises(Exception, match="blob missing"),
    ):
        GCSArtifactStore().load(DelayModel(), None)


def test_gcs_csv_client_read_raises_when_blob_not_found(monkeypatch):
    """read() propagates an exception when the GCS blob download fails."""
    monkeypatch.setenv("GCP_BUCKET_INPUT", "test-input-bucket")
    clear_settings_cache()

    mock_blob = MagicMock()
    mock_blob.download_as_bytes.side_effect = Exception("NotFound")
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    with (
        patch("google.cloud.storage.Client", return_value=mock_client),
        pytest.raises(Exception, match="NotFound"),
    ):
        GCSCSVClient(blob_name="serving_input.csv").read()

    clear_settings_cache()


def test_gcs_csv_client_write_creates_bigquery_client_when_none_injected():
    """write() instantiates a BigQueryClient when no _bq_writer was injected."""
    mock_writer_instance = MagicMock()
    mock_writer_cls = MagicMock(return_value=mock_writer_instance)
    identifiers = pd.DataFrame({"OPERA": ["Grupo LATAM"], "TIPOVUELO": ["I"], "MES": [6]})

    with patch("challenge.connectors.bigquery.BigQueryClient", mock_writer_cls):
        GCSCSVClient(blob_name="serving_input.csv").write([1], identifiers)

    mock_writer_cls.assert_called_once()
    mock_writer_instance.write.assert_called_once_with([1], identifiers)
