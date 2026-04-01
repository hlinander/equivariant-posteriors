"""
Ingestion process for reading unified Parquet files from staging (S3 or filesystem)
and splitting them into type-specific tables in the central database.

This script should be run periodically (e.g., every 5-10 minutes) to process
new data from training clients.

Usage:
    # One-time ingestion
    python ingestion/ingest.py

    # Continuous mode (recommended for production)
    python ingestion/ingest.py --continuous --interval 300

    # Dry run (test without modifying database)
    python ingestion/ingest.py --dry-run

The ingestion process uses AnalyticsConfig from env.py and automatically handles:
- Staging backend selection (S3 or filesystem)
- Central database configuration (DuckDB or DuckLake)
- S3 credentials and connection settings
"""

import argparse
import time
from pathlib import Path
import duckdb

# Table configuration
MODEL_PARAMETER = "model_parameter"
TRAIN_STEP_METRIC = "train_step_metric"
TRAIN_EPOCH_METRIC = "train_epoch_metric"
CHECKPOINT_SAMPLE_METRIC = "checkpoint_sample_metric"
MODELS_TABLE_NAME = "models"
RUNS_TABLE_NAME = "runs"
TRAIN_STEPS_TABLE_NAME = "train_steps"
CHECKPOINTS_TABLE_NAME = "checkpoints"
ARTIFACTS_TABLE_NAME = "artifacts"
ARTIFACT_CHUNKS_TABLE_NAME = "artifact_chunks"

SYNC_TABLES = [
    MODEL_PARAMETER,
    TRAIN_STEP_METRIC,
    TRAIN_EPOCH_METRIC,
    CHECKPOINT_SAMPLE_METRIC,
    MODELS_TABLE_NAME,
    RUNS_TABLE_NAME,
    TRAIN_STEPS_TABLE_NAME,
    CHECKPOINTS_TABLE_NAME,
    ARTIFACTS_TABLE_NAME,
    ARTIFACT_CHUNKS_TABLE_NAME,
]
TYPES = ["int", "float", "text"]
INGEST_BATCH_SIZE = 50
_SCHEMA_ENSURED = False


def get_s3_client(s3_key: str, s3_secret: str, s3_endpoint: str):
    """Create S3 client for file operations"""
    import boto3

    # Ensure endpoint has a protocol (boto3 requires it)
    if not s3_endpoint.startswith("http://") and not s3_endpoint.startswith("https://"):
        # Default to https if no protocol specified
        s3_endpoint = f"https://{s3_endpoint}"

    return boto3.client(
        "s3",
        aws_access_key_id=s3_key,
        aws_secret_access_key=s3_secret,
        endpoint_url=s3_endpoint,
    )


def ensure_s3_credentials(
    conn, s3_key: str, s3_secret: str, s3_region: str, s3_endpoint: str
):
    """Ensure S3 credentials are configured in DuckDB"""
    conn.execute("INSTALL aws")
    conn.execute("LOAD aws")

    # Configure S3 settings for MinIO compatibility
    # Use path-style URLs (endpoint/bucket/key) instead of virtual-hosted (bucket.endpoint/key)
    conn.execute("SET s3_url_style='path'")

    # Determine if we should use SSL based on endpoint
    use_ssl = not s3_endpoint.startswith("http://")
    conn.execute(f"SET s3_use_ssl={str(use_ssl).lower()}")

    # Strip protocol from endpoint - DuckDB adds it based on USE_SSL
    endpoint = s3_endpoint.replace("https://", "").replace("http://", "")

    conn.execute(
        f"""
        CREATE OR REPLACE SECRET (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{s3_key}',
            SECRET '{s3_secret}',
            REGION '{s3_region}',
            ENDPOINT '{endpoint}',
            URL_STYLE 'path',
            USE_SSL {str(use_ssl).lower()}
        )
        """
    )


def ensure_central_schema(conn):
    """
    Ensure all central database tables exist (idempotent).

    Creates the split schema where each table type (model_parameter, train_step_metric,
    checkpoint_sample_metric) is split into separate tables by value type (int, float, text).
    """
    # Model parameter tables (split by type)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_parameter_int (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value BIGINT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_parameter_float (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value FLOAT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_parameter_text (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            value TEXT
        )
    """
    )

    # Train step metric tables (split by type)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS train_step_metric_int (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            value BIGINT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS train_step_metric_float (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            value FLOAT
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS train_step_metric_text (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            name TEXT,
            step INTEGER,
            value TEXT
        )
    """
    )

    # Checkpoint sample metric tables (split by type)
    conn.execute(
        """
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
    """
    )
    conn.execute(
        """
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
    """
    )
    conn.execute(
        """
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
    """
    )

    # Train epoch metric table (no type splitting - explicit columns)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS train_epoch_metric (
            model_id BIGINT,
            run_id BIGINT,
            timestamp TIMESTAMPTZ,
            epoch INTEGER,
            step INTEGER,
            name TEXT,
            dataset TEXT,
            dataset_split TEXT,
            mean FLOAT,
            min FLOAT,
            max FLOAT,
            count INTEGER
        )
    """
    )

    # Models table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id BIGINT,
            train_id TEXT,
            timestamp TIMESTAMPTZ
        )
    """
    )

    # Runs table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id BIGINT,
            model_id BIGINT,
            timestamp TIMESTAMPTZ
        )
    """
    )

    # Train steps table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS train_steps (
            model_id BIGINT,
            run_id BIGINT,
            step INTEGER,
            dataset TEXT,
            sample_ids INTEGER[],
            timestamp TIMESTAMPTZ
        )
    """
    )

    # Checkpoints table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            model_id BIGINT,
            step INTEGER,
            path TEXT,
            timestamp TIMESTAMPTZ
        )
    """
    )

    # Artifacts table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id BIGINT,
            timestamp TIMESTAMPTZ,
            model_id BIGINT,
            name TEXT,
            path TEXT,
            type TEXT,
            size INTEGER
        )
    """
    )

    # Artifact chunks table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_chunks (
            artifact_id BIGINT,
            seq_num INTEGER,
            data BYTEA,
            size INTEGER,
            timestamp TIMESTAMPTZ
        )
    """
    )


def ensure_ingestion_state_table(conn):
    """Ensure the ingestion state tracking table exists"""
    # DuckLake doesn't support PRIMARY KEY, so we use a plain table
    # and handle duplicates manually in mark_file_processed
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_state (
            file_path TEXT,
            ingested_at TIMESTAMP,
            row_count INTEGER
        )
        """
    )


def get_processed_files(conn) -> set[str]:
    """Get set of already processed file paths"""
    result = conn.execute("SELECT DISTINCT file_path FROM ingestion_state").fetchall()
    return {row[0] for row in result}


def mark_file_processed(conn, file_path: str, row_count: int):
    """
    Mark a file as processed using INSERT OR IGNORE pattern.

    Since DuckLake doesn't support PRIMARY KEY, we check if the file
    already exists before inserting.
    """
    # Check if already processed
    exists = conn.execute(
        "SELECT 1 FROM ingestion_state WHERE file_path = ? LIMIT 1", (file_path,)
    ).fetchone()

    if not exists:
        conn.execute(
            """
            INSERT INTO ingestion_state (file_path, ingested_at, row_count)
            VALUES (?, now(), ?)
            """,
            (file_path, row_count),
        )


def list_s3_files(s3_client, bucket: str, prefix: str) -> list[str]:
    """List all parquet files in an S3 prefix"""
    files = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith(".parquet"):
                files.append(f"s3://{bucket}/{key}")

    return files


def move_s3_file(s3_client, bucket: str, source_key: str, dest_key: str):
    """Move a file within S3 (copy + delete)"""
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=dest_key,
    )
    s3_client.delete_object(Bucket=bucket, Key=source_key)


def list_filesystem_files(staging_dir: Path, table_name: str) -> list[str]:
    """List all parquet files in a filesystem directory for a given table"""
    table_dir = staging_dir / table_name
    if not table_dir.exists():
        return []

    files = []
    for file_path in table_dir.glob("*.parquet"):
        # Return absolute path as string
        files.append(str(file_path.absolute()))

    return files


def move_filesystem_file(source_path: Path, archive_dir: Path, table_name: str):
    """Move a file from staging to archive on filesystem"""
    import shutil

    # Create archive directory structure
    archive_table_dir = archive_dir / table_name
    archive_table_dir.mkdir(parents=True, exist_ok=True)

    # Destination path
    dest_path = archive_table_dir / source_path.name

    # Move file
    shutil.move(str(source_path), str(dest_path))
    print(f"[archive] Moved {source_path} -> {dest_path}")


def _pq_src(files: list[str]) -> str:
    """Build a read_parquet expression for a batch of files with filename tracking."""
    file_list = ", ".join(f"'{f}'" for f in files)
    return f"read_parquet([{file_list}], filename=true)"


def _ingest_batch(conn, table_name: str, batch: list[str]) -> int:
    """Ingest a batch of parquet files for a single table. Returns total rows."""
    pq = _pq_src(batch)

    if table_name in [MODEL_PARAMETER, TRAIN_STEP_METRIC, CHECKPOINT_SAMPLE_METRIC]:
        for type_name in TYPES:
            value_col = f"value_{type_name}"
            target_table = f"{table_name}_{type_name}"

            if table_name == MODEL_PARAMETER:
                conn.execute(
                    f"""
                    INSERT INTO {target_table} BY NAME
                    SELECT model_id, run_id, timestamp, name,
                           {value_col} as value
                    FROM {pq}
                    WHERE type = '{type_name}' AND {value_col} IS NOT NULL
                    """
                )
            elif table_name == TRAIN_STEP_METRIC:
                conn.execute(
                    f"""
                    INSERT INTO {target_table} BY NAME
                    SELECT model_id, run_id, timestamp, name, step,
                           {value_col} as value
                    FROM {pq}
                    WHERE type = '{type_name}' AND {value_col} IS NOT NULL
                    """
                )
            elif table_name == CHECKPOINT_SAMPLE_METRIC:
                mean_col = f"mean_{type_name}"
                vps_col = f"value_per_sample_{type_name}"
                conn.execute(
                    f"""
                    INSERT INTO {target_table} BY NAME
                    SELECT model_id, timestamp, step, name, dataset,
                           sample_ids,
                           {mean_col} as mean,
                           {vps_col} as value_per_sample
                    FROM {pq}
                    WHERE type = '{type_name}' AND {mean_col} IS NOT NULL
                    """
                )
    else:
        conn.execute(
            f"""
            INSERT INTO {table_name} BY NAME
            SELECT * EXCLUDE (filename) FROM {pq}
            """
        )

    # Get per-file row counts and bulk-insert into ingestion_state
    per_file_counts = dict(
        conn.execute(f"SELECT filename, COUNT(*) FROM {pq} GROUP BY filename").fetchall()
    )
    rows = [(f, per_file_counts.get(f, 0)) for f in batch]
    values = ", ".join(f"('{f}', now(), {count})" for f, count in rows)
    conn.execute(
        f"INSERT INTO ingestion_state (file_path, ingested_at, row_count) VALUES {values}"
    )

    return sum(count for _, count in rows)


def ingest_table(
    conn,
    table_name: str,
    parquet_files: list[str],
    processed_files: set[str],
    dry_run: bool = False,
    batch_size: int = INGEST_BATCH_SIZE,
) -> int:
    """
    Ingest a single unified table, splitting by type.

    Files are processed in batches to balance S3 round-trips against memory.

    Returns the number of files processed.
    """
    files_to_process = [f for f in parquet_files if f not in processed_files]

    if not files_to_process:
        print(f"[ingest] No new files to process for {table_name}")
        return 0

    print(f"[ingest] Processing {len(files_to_process)} files for {table_name}")

    if dry_run:
        for f in files_to_process:
            print(f"[ingest]   DRY RUN - would process {f}")
        return 0

    total_rows = 0
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i : i + batch_size]
        print(f"[ingest]   Batch {i // batch_size + 1}: {len(batch)} files")
        total_rows += _ingest_batch(conn, table_name, batch)

    print(
        f"[ingest] Ingested {total_rows} total rows from {len(files_to_process)} files"
    )
    return len(files_to_process)


def archive_processed_files(
    s3_client,
    bucket: str,
    staging_prefix: str,
    archive_prefix: str,
    processed_files: set[str],
):
    """Move processed files from staging to archive.

    Copies are parallelized with threads, then originals are batch-deleted.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Build list of (source_key, dest_key) pairs
    moves = []
    for s3_path in processed_files:
        if not s3_path.startswith(f"s3://{bucket}/{staging_prefix}"):
            continue

        key = s3_path.replace(f"s3://{bucket}/", "")
        filename = Path(key).name
        table_name = key.split("/")[1]
        archive_key = f"{archive_prefix}/{table_name}/{filename}"
        moves.append((key, archive_key))

    if not moves:
        return

    print(f"[archive] Archiving {len(moves)} files")

    # Parallel copy
    def copy_one(src_dest):
        src, dest = src_dest
        s3_client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": src},
            Key=dest,
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(copy_one, m): m for m in moves}
        done = 0
        for future in as_completed(futures):
            future.result()  # raise on error
            done += 1
            if done % 100 == 0 or done == len(moves):
                print(f"[archive]   Copied {done}/{len(moves)} files")

    # Batch delete originals
    source_keys = [src for src, _ in moves]
    for i in range(0, len(source_keys), 100):
        batch = source_keys[i : i + 100]
        s3_client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch]},
        )
        print(f"[archive]   Deleted batch {i // 100 + 1}: {len(batch)} files")

    print(f"[archive] Archived {len(moves)} files")


def ingest_all_from_config(config, dry_run: bool = False):
    """
    Ingest data using AnalyticsConfig

    This automatically:
    - Detects staging type (S3 vs filesystem)
    - Connects to the appropriate central database
    - Handles S3 credentials if needed

    Args:
        config: AnalyticsConfig instance (or None to load from env.py)
        dry_run: If True, don't actually modify the database
    """
    from lib.analytics_config import analytics_config

    if config is None:
        config = analytics_config()

    # Connect to central database based on config
    if config.is_ducklake_central():
        print("[ingest] Using DuckLake central database")
        conn = duckdb.connect()  # In-memory connection

        # Attach DuckLake
        postgres = config.central.postgres
        s3 = config.central.s3

        conn.execute("INSTALL ducklake")
        conn.execute("INSTALL aws")

        # Configure S3 for DuckLake data path
        ensure_s3_credentials(conn, s3.key, s3.secret, s3.region, s3.endpoint)

        print("[ingest] Attaching...")
        conn.execute(
            f"ATTACH IF NOT EXISTS 'ducklake:postgres:user={postgres.user} password={postgres.password} "
            f"host={postgres.host} port={postgres.port} dbname={postgres.dbname}' as central "
            f"(data_path '{config.central.data_path}')"
        )
        conn.execute("USE central")
        print("[ingest] Attached")

    elif config.is_duckdb_central():
        print(f"[ingest] Using local DuckDB database: {config.central.db_path}")
        conn = duckdb.connect(str(config.central.db_path))
    else:
        raise ValueError(f"Unknown central database type: {config.central.type}")

    # Ensure schema exists (once per process, not every ingestion cycle)
    global _SCHEMA_ENSURED
    if not _SCHEMA_ENSURED:
        print("[ingest] Ensuring central database schema exists")
        ensure_central_schema(conn)
        ensure_ingestion_state_table(conn)
        _SCHEMA_ENSURED = True

    # Get already processed files
    print("[ingest] Getting processed files...")
    processed_files = get_processed_files(conn)
    print(f"[ingest] Already processed {len(processed_files)} files")

    total_files = 0

    # Dispatch based on staging type
    if config.is_s3_staging():
        print(
            f"[ingest] Using S3 staging: s3://{config.staging.bucket}/{config.staging.prefix}"
        )

        s3 = config.staging.s3

        # Configure S3 credentials in DuckDB (for read_parquet)
        ensure_s3_credentials(conn, s3.key, s3.secret, s3.region, s3.endpoint)

        # Get S3 client for file operations (list/move)
        s3_client = get_s3_client(s3.key, s3.secret, s3.endpoint)

        # Process each table
        for table_name in SYNC_TABLES:
            table_prefix = f"{config.staging.prefix}/{table_name}"

            # List files in S3
            s3_files = list_s3_files(s3_client, config.staging.bucket, table_prefix)
            print(f"[ingest] Found {len(s3_files)} files for {table_name}")

            # Ingest
            files_processed = ingest_table(
                conn, table_name, s3_files, processed_files, dry_run
            )
            total_files += files_processed

            # Archive all staging files that have been ingested
            # (newly processed + any stragglers from prior runs)
            if not dry_run:
                archive_processed_files(
                    s3_client,
                    config.staging.bucket,
                    config.staging.prefix,
                    config.staging.archive_prefix,
                    set(s3_files),
                )

    elif config.is_filesystem_staging():
        print(f"[ingest] Using filesystem staging: {config.staging.staging_dir}")

        staging_dir = Path(config.staging.staging_dir)
        archive_dir = Path(config.staging.archive_dir)

        # Process each table
        for table_name in SYNC_TABLES:
            # List files in staging directory
            fs_files = list_filesystem_files(staging_dir, table_name)
            print(f"[ingest] Found {len(fs_files)} files for {table_name}")

            # Ingest (DuckDB's read_parquet works with filesystem paths)
            files_processed = ingest_table(
                conn, table_name, fs_files, processed_files, dry_run
            )
            total_files += files_processed

            # Archive all staging files that have been ingested
            if not dry_run:
                for file_path_str in fs_files:
                    file_path = Path(file_path_str)
                    move_filesystem_file(file_path, archive_dir, table_name)

    else:
        raise ValueError(f"Unknown staging type: {config.staging.type}")

    # Compact ingestion_state in DuckLake to merge small files
    if not dry_run and total_files > 0 and config.is_ducklake_central():
        print("[ingest] Compacting ingestion_state")
        conn.execute("CALL ducklake_merge_adjacent_files('central', 'ingestion_state')")

    print(f"[ingest] Ingestion complete. Processed {total_files} files.")
    conn.close()


def _ping_healthcheck(url: str, error: Exception | None = None):
    """Ping healthchecks.io to signal success or failure."""
    if not url:
        return
    import urllib.request

    try:
        if error is not None:
            ping_url = url.rstrip("/") + "/fail"
            body = str(error).encode("utf-8")
            req = urllib.request.Request(ping_url, data=body, method="POST")
        else:
            req = urllib.request.Request(url)
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[ingest] Health check ping failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest unified Parquet files from staging into central database using AnalyticsConfig from env.py"
    )

    parser.add_argument("--dry-run", action="store_true", help="Don't modify database")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (polling mode)",
    )

    args = parser.parse_args()

    # Load config from env.py
    from lib.analytics_config import analytics_config

    config = analytics_config()

    print("[ingest] Using AnalyticsConfig from env.py")
    print(f"[ingest]   Staging: {config.staging.type}")
    print(f"[ingest]   Central: {config.central.type}")

    if args.continuous:
        interval = config.ingest_interval_seconds
        print(
            f"[ingest] Running in continuous mode (interval: {interval}s from config)"
        )
        if config.healthcheck_url:
            print(f"[ingest] Health check enabled: {config.healthcheck_url}")
        while True:
            try:
                ingest_all_from_config(config, dry_run=args.dry_run)
                _ping_healthcheck(config.healthcheck_url)
            except Exception as e:
                print(f"[ingest] Error during ingestion: {e}")
                import traceback

                traceback.print_exc()
                _ping_healthcheck(config.healthcheck_url, error=e)

            print(f"[ingest] Sleeping for {interval} seconds...")
            time.sleep(interval)
    else:
        ingest_all_from_config(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
