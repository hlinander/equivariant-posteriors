"""
Test the full ingestion pipeline: export → S3 → ingestion → central DB

No external services required! Run with:
    uv run --extra test pytest test/test_ingestion.py -v

Uses moto server for S3 (in-memory, no Docker needed).
"""
import sys
import time
import pytest
import duckdb
from pathlib import Path

import boto3

from lib.train import create_initial_state
import lib.render_duck as duck
from lib.export import export_all

# Import create_train_run from test/conftest.py
sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


BUCKET_NAME = "test-metrics-staging"
MOTO_PORT = 5556  # Different port from test_s3_export to avoid conflicts
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
def test_env(moto_server, tmp_path):
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

    central_db_path = tmp_path / "central.db"

    analytics_config_module._analytics_config = AnalyticsConfig(
        staging=StagingS3(
            s3=s3_config,
            bucket=BUCKET_NAME,
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckDB(db_path=central_db_path),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    # Clean up any existing DuckDB connection
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    yield {
        "s3_client": s3_client,
        "central_db_path": central_db_path,
        "config": analytics_config_module._analytics_config,
    }

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


def test_full_pipeline(test_env):
    """Test the complete pipeline: export → S3 → ingestion → central DB"""
    s3_client = test_env["s3_client"]
    central_db_path = test_env["central_db_path"]
    config = test_env["config"]

    # Step 1: Create client data and export to S3
    print("\n[test] Step 1: Creating client data and exporting to S3")

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

    # Export to S3 using AnalyticsConfig
    paths = export_all(train_run)

    print(f"[test] Exported {len(paths)} files to S3:")
    for path in paths:
        print(f"  - {path}")

    assert len(paths) > 0, "Should have exported files"

    # Step 2: Run ingestion (schema will be created automatically)
    print("\n[test] Step 2: Running ingestion")

    from ingestion.ingest import ingest_all_from_config

    ingest_all_from_config(config, dry_run=False)

    # Step 3: Verify data in central database
    print("\n[test] Step 3: Verifying data in central database")

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

    # Step 4: Verify idempotency (run ingestion again, should not duplicate)
    print("\n[test] Step 4: Testing idempotency")

    ingest_all_from_config(config, dry_run=False)

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


def test_incremental_ingestion(test_env):
    """Test that ingestion picks up new files incrementally"""
    s3_client = test_env["s3_client"]
    central_db_path = test_env["central_db_path"]
    config = test_env["config"]

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

    from ingestion.ingest import ingest_all_from_config

    ingest_all_from_config(config, dry_run=False)

    # Step 3: Add more data and export
    print("\n[test] Step 3: Second export with new data")
    duck.insert_model_parameter(model_id, train_run.run_id, "param2", 200)

    paths2 = export_all(train_run)
    print(f"[test] Second export: {len(paths2)} files")

    # Step 4: Second ingestion (should only process new file)
    print("\n[test] Step 4: Second ingestion")
    ingest_all_from_config(config, dry_run=False)

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
    pytest.main([__file__, "-v"])
