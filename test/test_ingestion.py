"""
Test the full ingestion pipeline: export → MinIO → ingestion → central DB

Prerequisites:
    1. Start MinIO: docker-compose up -d
    2. Run: pytest test/test_ingestion.py

This test verifies the complete flow from client export to central database using AnalyticsConfig.
"""
import sys
import time
import pytest
import duckdb
from pathlib import Path

from lib.train import create_initial_state
import lib.render_duck as duck
from lib.export import export_all

# Import create_train_run from test/conftest.py
sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


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
def test_env():
    """Fixture to set up and tear down test environment"""
    setup_local_env()

    # Clean MinIO staging area before test
    clear_minio_staging()

    # Clean up any existing connection
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


def test_full_pipeline(test_env, tmp_path):
    """Test the complete pipeline: export → MinIO → ingestion → central DB"""

    # Step 1: Create client data and export to MinIO
    print("\n[test] Step 1: Creating client data and exporting to MinIO")

    duck.ensure_duck(None, True)  # In-memory client DB

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = state.model_id  # create_initial_state already calls insert_model

    # Insert various types of metrics
    duck.insert_model_parameter(model_id, train_run.run_id, "learning_rate", 0.001)  # float
    duck.insert_model_parameter(model_id, train_run.run_id, "batch_size", 32)  # int
    duck.insert_model_parameter(model_id, train_run.run_id, "optimizer", "adam")  # text

    for i in range(10):
        duck.insert_train_step_metric(model_id, train_run.run_id, "loss", i, 0.5 - i * 0.01)  # float
        duck.insert_train_step_metric(model_id, train_run.run_id, "epoch", i, i)  # int

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

    # Insert run
    duck.insert_run(train_run.run_id, model_id)

    # Insert train steps
    train_sample_ids = [10, 11, 12, 13, 14]
    for i in range(3):
        duck.insert_train_step(model_id, train_run.run_id, i, "train_dataset", train_sample_ids)

    # Insert checkpoint
    duck.insert_checkpoint(model_id, 100, "/path/to/checkpoint.pt")

    # Export to S3/MinIO using AnalyticsConfig
    paths = export_all(train_run)

    print(f"[test] Exported {len(paths)} files to MinIO:")
    for path in paths:
        print(f"  - {path}")

    assert len(paths) > 0, "Should have exported files"

    # Step 2: Update AnalyticsConfig to use the test central DB
    # (Schema will be created automatically by ingestion script)
    from lib.analytics_config import AnalyticsConfig, StagingS3, CentralDuckDB, S3Config
    import lib.analytics_config as analytics_config_module

    central_db_path = tmp_path / "central.db"

    s3_minio = S3Config(
        key="minioadmin",
        secret="minioadmin",
        region="us-east-1",
        endpoint="http://localhost:9000",
    )

    test_config = AnalyticsConfig(
        staging=StagingS3(
            s3=s3_minio,
            bucket="metrics-staging",
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckDB(db_path=central_db_path),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    # Step 3: Run ingestion (schema will be created automatically)
    print("\n[test] Step 3: Running ingestion")

    from ingestion.ingest import ingest_all_from_config

    ingest_all_from_config(test_config, dry_run=False)

    # Step 4: Verify data in central database
    print("\n[test] Step 4: Verifying data in central database")

    # Reconnect to see the changes
    central_conn = duckdb.connect(str(central_db_path))

    # Check model parameters
    float_params = central_conn.execute(
        "SELECT name, value FROM model_parameter_float WHERE name = 'learning_rate'"
    ).fetchall()
    assert len(float_params) == 1
    assert float_params[0][0] == "learning_rate"
    assert abs(float_params[0][1] - 0.001) < 1e-6
    print(f"[test] ✓ Float parameter: {float_params[0]}")

    int_params = central_conn.execute(
        "SELECT name, value FROM model_parameter_int WHERE name = 'batch_size'"
    ).fetchall()
    assert len(int_params) == 1
    assert int_params[0][1] == 32
    print(f"[test] ✓ Int parameter: {int_params[0]}")

    text_params = central_conn.execute(
        "SELECT name, value FROM model_parameter_text WHERE name = 'optimizer'"
    ).fetchall()
    assert len(text_params) == 1
    assert text_params[0][1] == "adam"
    print(f"[test] ✓ Text parameter: {text_params[0]}")

    # Check train step metrics
    loss_metrics = central_conn.execute(
        "SELECT COUNT(*) FROM train_step_metric_float WHERE name = 'loss'"
    ).fetchone()
    assert loss_metrics[0] == 10
    print(f"[test] ✓ Found {loss_metrics[0]} loss metrics")

    epoch_metrics = central_conn.execute(
        "SELECT COUNT(*) FROM train_step_metric_int WHERE name = 'epoch'"
    ).fetchone()
    assert epoch_metrics[0] == 10
    print(f"[test] ✓ Found {epoch_metrics[0]} epoch metrics")

    # Check checkpoint sample metrics
    checkpoint_metrics = central_conn.execute(
        "SELECT COUNT(*) FROM checkpoint_sample_metric_float WHERE name = 'val_accuracy'"
    ).fetchone()
    assert checkpoint_metrics[0] == 1
    print(f"[test] ✓ Found {checkpoint_metrics[0]} checkpoint sample metric")

    # Verify the array values
    checkpoint_data = central_conn.execute(
        "SELECT mean, value_per_sample FROM checkpoint_sample_metric_float WHERE name = 'val_accuracy'"
    ).fetchone()
    assert abs(checkpoint_data[0] - sum(values) / len(values)) < 1e-6
    # Check array values with approximate equality (FLOAT precision loss)
    assert len(checkpoint_data[1]) == len(values)
    for i, (actual, expected) in enumerate(zip(checkpoint_data[1], values)):
        assert abs(actual - expected) < 1e-6, f"Value {i}: {actual} != {expected}"
    print(f"[test] ✓ Checkpoint metric mean: {checkpoint_data[0]}")
    print(f"[test] ✓ Checkpoint metric values: {checkpoint_data[1]}")

    # Check models table
    models_count = central_conn.execute(
        "SELECT COUNT(*) FROM models"
    ).fetchone()
    assert models_count[0] == 1, f"Expected 1 model, got {models_count[0]}"
    print(f"[test] ✓ Found {models_count[0]} model in models table")

    # Check runs table
    runs_count = central_conn.execute(
        "SELECT COUNT(*) FROM runs"
    ).fetchone()
    assert runs_count[0] == 1
    print(f"[test] ✓ Found {runs_count[0]} run in runs table")

    # Check train_steps table
    train_steps_count = central_conn.execute(
        "SELECT COUNT(*) FROM train_steps"
    ).fetchone()
    assert train_steps_count[0] == 3
    print(f"[test] ✓ Found {train_steps_count[0]} train steps in train_steps table")

    # Check checkpoints table
    checkpoints_count = central_conn.execute(
        "SELECT COUNT(*) FROM checkpoints"
    ).fetchone()
    assert checkpoints_count[0] == 1
    checkpoint_path = central_conn.execute(
        "SELECT path FROM checkpoints"
    ).fetchone()
    assert checkpoint_path[0] == "/path/to/checkpoint.pt"
    print(f"[test] ✓ Found {checkpoints_count[0]} checkpoint in checkpoints table")

    # Check artifacts table (should be 0 for this test)
    artifacts_count = central_conn.execute(
        "SELECT COUNT(*) FROM artifacts"
    ).fetchone()
    print(f"[test] ✓ Found {artifacts_count[0]} artifacts in artifacts table")

    # Check artifact_chunks table (should be 0 for this test)
    artifact_chunks_count = central_conn.execute(
        "SELECT COUNT(*) FROM artifact_chunks"
    ).fetchone()
    print(f"[test] ✓ Found {artifact_chunks_count[0]} artifact chunks in artifact_chunks table")

    # Step 5: Verify idempotency (run ingestion again, should not duplicate)
    print("\n[test] Step 5: Testing idempotency")

    ingest_all_from_config(test_config, dry_run=False)

    # Reconnect and verify counts didn't change
    central_conn = duckdb.connect(str(central_db_path))

    float_params_count = central_conn.execute(
        "SELECT COUNT(*) FROM model_parameter_float WHERE name = 'learning_rate'"
    ).fetchone()
    assert float_params_count[0] == 1, "Should not duplicate on re-ingestion"

    # Also check one of the new tables
    runs_count_after = central_conn.execute(
        "SELECT COUNT(*) FROM runs"
    ).fetchone()
    assert runs_count_after[0] == 1, "Should not duplicate runs on re-ingestion"

    train_steps_count_after = central_conn.execute(
        "SELECT COUNT(*) FROM train_steps"
    ).fetchone()
    assert train_steps_count_after[0] == 3, "Should not duplicate train_steps on re-ingestion"

    print(f"[test] ✓ No duplicates after re-ingestion (checked parameters, runs, and train_steps)")

    central_conn.close()
    print("\n[test] ✅ Full pipeline test passed!")


def test_incremental_ingestion(test_env, tmp_path):
    """Test that ingestion picks up new files incrementally"""

    # Step 1: First export
    print("\n[test] Step 1: First export")
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = state.model_id  # create_initial_state already calls insert_model

    duck.insert_model_parameter(model_id, train_run.run_id, "param1", 100)

    paths1 = export_all(train_run)
    print(f"[test] First export: {len(paths1)} files")

    # Step 2: First ingestion (schema will be created automatically)
    print("\n[test] Step 2: First ingestion")
    central_db_path = tmp_path / "central.db"

    from lib.analytics_config import AnalyticsConfig, StagingS3, CentralDuckDB, S3Config
    from ingestion.ingest import ingest_all_from_config

    s3_minio = S3Config(
        key="minioadmin",
        secret="minioadmin",
        region="us-east-1",
        endpoint="http://localhost:9000",
    )

    test_config = AnalyticsConfig(
        staging=StagingS3(
            s3=s3_minio,
            bucket="metrics-staging",
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckDB(db_path=central_db_path),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    ingest_all_from_config(test_config, dry_run=False)

    # Step 3: Add more data and export
    print("\n[test] Step 3: Second export with new data")
    duck.insert_model_parameter(model_id, train_run.run_id, "param2", 200)

    paths2 = export_all(train_run)
    print(f"[test] Second export: {len(paths2)} files")

    # Step 4: Second ingestion (should only process new file)
    print("\n[test] Step 4: Second ingestion")
    ingest_all_from_config(test_config, dry_run=False)

    # Verify both parameters are present
    central_conn = duckdb.connect(str(central_db_path))
    param_count = central_conn.execute(
        "SELECT COUNT(*) FROM model_parameter_int"
    ).fetchone()
    assert param_count[0] == 2, f"Should have 2 parameters, got {param_count[0]}"
    print(f"[test] ✓ Found {param_count[0]} parameters after incremental ingestion")

    central_conn.close()
    print("\n[test] ✅ Incremental ingestion test passed!")


if __name__ == "__main__":
    # For manual testing
    import tempfile

    setup_local_env()
    print("Starting ingestion pipeline tests...")
    print("Make sure MinIO is running: docker-compose up -d")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Clean up connection
        if duck.CONN is not None:
            duck.CONN.close()
            duck.CONN = None
            duck.SCHEMA_ENSURED = False

        test_full_pipeline(None, tmp_path)
        print("\n" + "="*60 + "\n")

        # Clean up connection
        if duck.CONN is not None:
            duck.CONN.close()
            duck.CONN = None
            duck.SCHEMA_ENSURED = False

        test_incremental_ingestion(None, tmp_path)

    print("\nAll ingestion tests passed!")
