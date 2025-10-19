"""
Test filesystem staging functionality

This test suite verifies the complete pipeline using filesystem staging
instead of S3. Perfect for environments without S3 access (e.g., HPC with NFS/Lustre).

No external dependencies required - runs entirely on local filesystem!
"""
import sys
import pytest
import duckdb
from pathlib import Path

from lib.train import create_initial_state
import lib.render_duck as duck
from lib.export import export_all

# Import create_train_run from test/conftest.py
sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_train_run


def setup_filesystem_env(staging_dir: Path, central_db_path: Path):
    """Set up AnalyticsConfig for filesystem staging"""
    from lib.analytics_config import AnalyticsConfig, StagingFilesystem, CentralDuckDB
    import lib.analytics_config as analytics_config_module

    analytics_config_module._analytics_config = AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=staging_dir,
            archive_dir=staging_dir.parent / "archive",
        ),
        central=CentralDuckDB(db_path=central_db_path),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    print(f"[test] Using filesystem staging: {staging_dir}")


@pytest.fixture
def filesystem_env(tmp_path):
    """Fixture to set up and tear down filesystem staging environment"""
    staging_dir = tmp_path / "staging"
    central_db = tmp_path / "central.db"

    staging_dir.mkdir(parents=True, exist_ok=True)

    setup_filesystem_env(staging_dir, central_db)

    # Clean up any existing connection before test
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    yield tmp_path

    # Cleanup after test
    if duck.CONN is not None:
        duck.CONN.close()
        duck.CONN = None
        duck.SCHEMA_ENSURED = False

    # Reset analytics config
    import lib.analytics_config as analytics_config_module
    analytics_config_module._analytics_config = None


def test_export_to_filesystem(filesystem_env):
    """Test exporting metrics to local filesystem"""
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

    # All paths should be filesystem paths (not S3)
    for path in paths:
        assert not str(path).startswith("s3://"), f"Path should be filesystem path: {path}"
        # Verify files actually exist
        file_path = Path(path)
        assert file_path.exists(), f"File should exist: {path}"
        assert file_path.stat().st_size > 0, f"File should not be empty: {path}"


def test_incremental_export_filesystem(filesystem_env):
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


def create_central_db_schema(conn):
    """Create the split schema in central database (type-specific tables)"""
    # Model parameter tables (split by type)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_parameter_int (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value BIGINT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_parameter_float (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value FLOAT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_parameter_text (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value TEXT
        )
    """)

    # Train step metric tables (split by type)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS train_step_metric_int (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            value BIGINT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS train_step_metric_float (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            value FLOAT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS train_step_metric_text (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            value TEXT
        )
    """)

    # Checkpoint sample metric tables (split by type)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_sample_metric_int (
            model_id BIGINT,
            timestamp TIMESTAMPTZ,
            step INTEGER,
            name TEXT,
            dataset TEXT,
            sample_ids INTEGER[],
            mean BIGINT,
            value_per_sample BIGINT[]
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_sample_metric_float (
            model_id BIGINT,
            timestamp TIMESTAMPTZ,
            step INTEGER,
            name TEXT,
            dataset TEXT,
            sample_ids INTEGER[],
            mean FLOAT,
            value_per_sample FLOAT[]
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_sample_metric_text (
            model_id BIGINT,
            timestamp TIMESTAMPTZ,
            step INTEGER,
            name TEXT,
            dataset TEXT,
            sample_ids INTEGER[],
            mean TEXT,
            value_per_sample TEXT[]
        )
    """)


def test_full_filesystem_pipeline(filesystem_env):
    """Test the complete pipeline: export → filesystem → ingestion → central DB"""

    # Step 1: Create client data and export to filesystem
    print("\n[test] Step 1: Creating client data and exporting to filesystem")

    duck.ensure_duck(None, True)  # In-memory client DB

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

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

    # Export to filesystem using AnalyticsConfig
    paths = export_all(train_run)

    print(f"[test] Exported {len(paths)} files to filesystem:")
    for path in paths:
        print(f"  - {path}")

    assert len(paths) > 0, "Should have exported files"

    # Step 2: Create central database
    print("\n[test] Step 2: Creating central database")

    central_db_path = filesystem_env / "central.db"
    central_conn = duckdb.connect(str(central_db_path))
    create_central_db_schema(central_conn)
    central_conn.close()

    # Step 3: Update AnalyticsConfig to use the test central DB
    from lib.analytics_config import AnalyticsConfig, StagingFilesystem, CentralDuckDB
    import lib.analytics_config as analytics_config_module

    test_config = AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=filesystem_env / "staging",
            archive_dir=filesystem_env / "archive",
        ),
        central=CentralDuckDB(db_path=central_db_path),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    # Step 4: Run ingestion
    print("\n[test] Step 4: Running ingestion")

    from ingestion.ingest import ingest_all_from_config

    ingest_all_from_config(test_config, dry_run=False)

    # Step 5: Verify data in central database
    print("\n[test] Step 5: Verifying data in central database")

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

    # Step 6: Verify idempotency (run ingestion again, should not duplicate)
    print("\n[test] Step 6: Testing idempotency")

    ingest_all_from_config(test_config, dry_run=False)

    # Reconnect and verify counts didn't change
    central_conn = duckdb.connect(str(central_db_path))

    float_params_count = central_conn.execute(
        "SELECT COUNT(*) FROM model_parameter_float WHERE name = 'learning_rate'"
    ).fetchone()
    assert float_params_count[0] == 1, "Should not duplicate on re-ingestion"
    print(f"[test] ✓ No duplicates after re-ingestion")

    # Step 7: Verify files were moved to archive
    print("\n[test] Step 7: Verifying archive")

    archive_dir = filesystem_env / "archive"
    assert archive_dir.exists(), "Archive directory should exist"

    # Count archived files
    archived_files = list(archive_dir.rglob("*.parquet"))
    print(f"[test] ✓ Found {len(archived_files)} files in archive")
    assert len(archived_files) == len(paths), "All exported files should be archived"

    # Verify staging is empty (files moved to archive)
    staging_dir = filesystem_env / "staging"
    staging_files = list(staging_dir.rglob("*.parquet"))
    print(f"[test] ✓ Found {len(staging_files)} files in staging (should be 0)")
    assert len(staging_files) == 0, "Staging should be empty after ingestion"

    central_conn.close()
    print("\n[test] ✅ Full filesystem pipeline test passed!")


def test_incremental_ingestion_filesystem(filesystem_env):
    """Test that filesystem ingestion picks up new files incrementally"""

    # Step 1: First export
    print("\n[test] Step 1: First export")
    duck.ensure_duck(None, True)

    train_run = create_train_run()
    state = create_initial_state(train_run, None, "cpu")
    model_id = duck.insert_model(train_run)

    duck.insert_model_parameter(model_id, train_run.run_id, "param1", 100)

    paths1 = export_all(train_run)
    print(f"[test] First export: {len(paths1)} files")

    # Step 2: First ingestion
    print("\n[test] Step 2: First ingestion")
    central_db_path = filesystem_env / "central.db"
    central_conn = duckdb.connect(str(central_db_path))
    create_central_db_schema(central_conn)
    central_conn.close()

    from lib.analytics_config import AnalyticsConfig, StagingFilesystem, CentralDuckDB
    from ingestion.ingest import ingest_all_from_config

    test_config = AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=filesystem_env / "staging",
            archive_dir=filesystem_env / "archive",
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
    print("\n[test] ✅ Incremental filesystem ingestion test passed!")


if __name__ == "__main__":
    # For manual testing
    import tempfile

    print("Starting filesystem staging tests...")
    print("No external dependencies needed - runs entirely on local filesystem!")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Clean up connection
        if duck.CONN is not None:
            duck.CONN.close()
            duck.CONN = None
            duck.SCHEMA_ENSURED = False

        test_export_to_filesystem(tmp_path)
        print("\n" + "="*60 + "\n")

        # Clean up connection
        if duck.CONN is not None:
            duck.CONN.close()
            duck.CONN = None
            duck.SCHEMA_ENSURED = False

        test_full_filesystem_pipeline(tmp_path)

    print("\nAll filesystem staging tests passed!")
