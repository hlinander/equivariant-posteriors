"""
Test S3 export functionality using moto server (in-memory S3)

No external services required! Run with:
    uv run --extra test pytest test/test_s3_export.py -v

Moto runs as a real HTTP server that DuckDB can connect to.
"""
import time
import pytest
from pathlib import Path

import boto3

# Import from root test.py (not test/ package)
import importlib.util
_test_utils_path = Path(__file__).parent.parent / "test.py"
_spec = importlib.util.spec_from_file_location("test_utils", _test_utils_path)
_test_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_test_utils)
create_train_run = _test_utils.create_train_run

from lib.train import create_initial_state
import lib.render_duck as duck
from lib.export import export_all


BUCKET_NAME = "test-metrics-staging"
MOTO_PORT = 5555
MOTO_ENDPOINT = f"http://127.0.0.1:{MOTO_PORT}"


@pytest.fixture(scope="module")
def moto_server():
    """Start moto server for the test module"""
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=MOTO_PORT, verbose=False)
    server.start()

    # Wait for server to be ready
    import urllib.request
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{MOTO_ENDPOINT}/moto-api/")
            break
        except Exception:
            time.sleep(0.1)

    yield server

    server.stop()


@pytest.fixture
def s3_env(moto_server):
    """Fixture to set up S3 environment with moto server"""
    # Create S3 client pointing to moto server
    s3_client = boto3.client(
        "s3",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        endpoint_url=MOTO_ENDPOINT,
        region_name="us-east-1",
    )

    # Create bucket
    try:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        pass

    # Configure AnalyticsConfig to use moto server
    from lib.analytics_config import AnalyticsConfig, StagingS3, CentralDuckDB, S3Config
    import lib.analytics_config as analytics_config_module

    s3_config = S3Config(
        key="testing",
        secret="testing",
        region="us-east-1",
        endpoint=MOTO_ENDPOINT,
    )

    analytics_config_module._analytics_config = AnalyticsConfig(
        staging=StagingS3(
            s3=s3_config,
            bucket=BUCKET_NAME,
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckDB(db_path=Path("./test_central.db")),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    # Clean up any existing DuckDB connection
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    yield s3_client

    # Cleanup after test - clear bucket
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
        if "Contents" in response:
            objects = [{"Key": obj["Key"]} for obj in response["Contents"]]
            if objects:
                s3_client.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": objects})
    except Exception:
        pass

    # Cleanup DuckDB
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    analytics_config_module._analytics_config = None


def list_s3_files(s3_client, bucket: str, prefix: str = "") -> list[str]:
    """List all files in S3"""
    files = []
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" in response:
        for obj in response["Contents"]:
            files.append(f"s3://{bucket}/{obj['Key']}")
    return files


def test_export_to_s3(s3_env):
    """Test exporting metrics to moto S3 server"""
    s3_client = s3_env

    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    for i in range(10):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", i, 0.5 - i * 0.01
        )
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "accuracy", i, 0.9 + i * 0.01
        )

    duck.insert_model_parameter(model_id, train_run.run_id, "learning_rate", 0.001)
    duck.insert_model_parameter(model_id, train_run.run_id, "batch_size", 32)
    duck.insert_model_parameter(model_id, train_run.run_id, "model_type", "test")

    paths = export_all(train_run)

    print(f"\n[test] Exported {len(paths)} files:")
    for path in paths:
        print(f"  - {path}")

    assert len(paths) > 0, "Should have exported at least one file"

    for path in paths:
        assert path.startswith(f"s3://{BUCKET_NAME}/"), f"Path should be in bucket: {path}"

    s3_files = list_s3_files(s3_client, BUCKET_NAME, "staging/")
    assert len(s3_files) > 0, "Should have files in S3"


def test_incremental_export(s3_env):
    """Test that only new data is exported on subsequent runs"""
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    for i in range(5):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", i, 0.5
        )

    paths1 = export_all(train_run)
    print(f"\n[test] First export: {len(paths1)} files")

    paths2 = export_all(train_run)
    print(f"[test] Second export (no new data): {len(paths2)} files")
    assert len(paths2) == 0, "Should not export anything when no new data"

    for i in range(5, 10):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", i, 0.4
        )

    paths3 = export_all(train_run)
    print(f"[test] Third export (new data): {len(paths3)} files")
    assert len(paths3) > 0, "Should export new data"


def test_export_with_checkpoint_metrics(s3_env):
    """Test exporting checkpoint sample metrics"""
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    sample_ids = [1, 2, 3, 4, 5]
    values = [0.9, 0.85, 0.92, 0.88, 0.91]

    duck.insert_checkpoint_sample_metric(
        model_id=model_id,
        step=100,
        name="val_accuracy",
        dataset="test_dataset",
        sample_ids=sample_ids,
        mean=sum(values) / len(values),
        value_per_sample=values,
    )

    paths = export_all(train_run)

    print(f"\n[test] Exported checkpoint metrics: {len(paths)} files")
    assert len(paths) > 0, "Should have exported checkpoint metrics"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
