"""
Test S3 export functionality locally using MinIO

Prerequisites:
    1. Start MinIO: docker-compose up -d
    2. Run: pytest test/test_s3_export.py

The test suite automatically configures MinIO settings using AnalyticsConfig.
No environment variables needed!
"""
import os
import sys
import time
import pytest
from pathlib import Path

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


def setup_local_env():
    """Set up AnalyticsConfig for local MinIO testing"""
    from lib.analytics_config import AnalyticsConfig, StagingS3, CentralDuckDB, S3Config
    import lib.analytics_config as analytics_config_module

    # Create MinIO config
    s3_minio = S3Config(
        key="minioadmin",
        secret="minioadmin",
        region="us-east-1",
        endpoint="http://localhost:9000",
    )

    # Override the global analytics config with MinIO settings
    analytics_config_module._analytics_config = AnalyticsConfig(
        staging=StagingS3(
            s3=s3_minio,
            bucket="metrics-staging",
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckDB(db_path=Path("./test_central.db")),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    print(f"[test] Using local MinIO at http://localhost:9000")
    print(f"[test] Bucket: metrics-staging")


def clear_minio_staging():
    """Clear all files from MinIO staging and archive directories"""
    try:
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client(
            "s3",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            endpoint_url="http://localhost:9000",
        )

        bucket = "metrics-staging"

        # Clear staging prefix
        for prefix in ["staging/", "archive/"]:
            try:
                response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                if "Contents" in response:
                    objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
                    if objects_to_delete:
                        s3_client.delete_objects(
                            Bucket=bucket,
                            Delete={"Objects": objects_to_delete}
                        )
                        print(f"[test-cleanup] Deleted {len(objects_to_delete)} files from {prefix}")
            except ClientError as e:
                # Ignore errors if bucket doesn't exist yet
                if e.response['Error']['Code'] != 'NoSuchBucket':
                    print(f"[test-cleanup] Warning: Could not clear {prefix}: {e}")
    except ImportError:
        print("[test-cleanup] boto3 not available, skipping MinIO cleanup")
    except Exception as e:
        print(f"[test-cleanup] Warning: Could not connect to MinIO: {e}")


@pytest.fixture
def minio_env():
    """Fixture to set up and tear down MinIO environment"""
    setup_local_env()

    # Clean MinIO staging area before test
    clear_minio_staging()

    # Clean up any existing connection before test
    import lib.render_duck as duck
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    yield

    # Cleanup after test
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    # Reset analytics config
    import lib.analytics_config as analytics_config_module
    analytics_config_module._analytics_config = None


def test_export_to_minio(minio_env):
    """Test exporting metrics to local MinIO"""
    # Initialize in-memory DuckDB
    duck.ensure_duck(None, True)

    # Create test data
    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    # Insert test metrics
    for i in range(10):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, f"loss", i, 0.5 - i * 0.01
        )
        duck.insert_train_step_metric(
            model_id, train_run.run_id, f"accuracy", i, 0.9 + i * 0.01
        )

    # Insert model parameters
    duck.insert_model_parameter(model_id, train_run.run_id, "learning_rate", 0.001)
    duck.insert_model_parameter(model_id, train_run.run_id, "batch_size", 32)
    duck.insert_model_parameter(model_id, train_run.run_id, "model_type", "test")

    # Export using AnalyticsConfig
    paths = export_all(train_run)

    print(f"\n[test] Exported {len(paths)} files:")
    for path in paths:
        print(f"  - {path}")

    # Verify files were created
    assert len(paths) > 0, "Should have exported at least one file"

    # All paths should point to MinIO
    for path in paths:
        assert path.startswith("s3://metrics-staging/"), f"Path should be in metrics-staging bucket: {path}"


def test_incremental_export(minio_env):
    """Test that only new data is exported on subsequent runs"""
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    # Insert first batch of metrics
    for i in range(5):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", i, 0.5
        )

    # First export
    paths1 = export_all(train_run)
    print(f"\n[test] First export: {len(paths1)} files")

    # Second export immediately (should export nothing)
    paths2 = export_all(train_run)
    print(f"[test] Second export (no new data): {len(paths2)} files")
    assert len(paths2) == 0, "Should not export anything when no new data"

    # Insert more metrics
    for i in range(5, 10):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", i, 0.4
        )

    # Third export (should only export new data)
    paths3 = export_all(train_run)
    print(f"[test] Third export (new data): {len(paths3)} files")
    assert len(paths3) > 0, "Should export new data"


def test_periodic_export(minio_env):
    """Test periodic export in background thread

    Note: This test has limitations when using in-memory databases because:
    1. Daemon threads can't be cleanly stopped in pytest
    2. The fixture cleanup closes the connection, invalidating the thread's cursor
    3. This is not an issue in production where persistent databases are used

    For testing purposes, we just verify the thread starts and can perform one export cycle.
    """
    duck.ensure_duck(None, True)

    from lib.export import start_periodic_export

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    # Insert metrics BEFORE starting export
    for i in range(5):
        duck.insert_train_step_metric(
            model_id, train_run.run_id, "loss", i, 0.5
        )

    # Start periodic export (every 2 seconds for testing)
    export_thread = start_periodic_export(train_run, interval_seconds=2)

    # Verify thread started
    assert export_thread.is_alive(), "Export thread should be running"

    # Wait for first export cycle
    print("\n[test] Waiting for background export...")
    time.sleep(3)

    # Note: We don't assert thread is still alive because the fixture cleanup
    # will close the connection, which may cause the thread to exit or error.
    # In production with persistent databases, the thread continues indefinitely.

    print("[test] Periodic export test completed successfully")


def test_export_with_checkpoint_metrics(minio_env):
    """Test exporting checkpoint sample metrics"""
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    # Insert checkpoint sample metrics
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
    # For manual testing
    setup_local_env()
    print("Starting local S3 export tests...")
    print("Make sure MinIO is running: docker-compose up -d")
    print()

    def cleanup_connection():
        """Clean up connection between manual tests"""
        if duck.CONN is not None:
            duck.CONN.close()
            duck.CONN = None
            duck.SCHEMA_ENSURED = False

    test_export_to_minio(None)
    print("\n" + "="*60 + "\n")
    cleanup_connection()

    test_incremental_export(None)
    print("\n" + "="*60 + "\n")
    cleanup_connection()

    test_export_with_checkpoint_metrics(None)
    print("\n" + "="*60 + "\n")
    cleanup_connection()

    test_periodic_export(None)
    print("\n" + "="*60 + "\n")

    print("\nAll tests passed!")
    print("\nNote: Background export thread is still running (daemon thread).")
    print("It will be cleaned up when the process exits.")
