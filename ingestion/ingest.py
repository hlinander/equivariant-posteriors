"""
Ingestion process for reading unified Parquet files from staging (S3 or filesystem)
and splitting them into type-specific tables in the central database.

This script should be run periodically (e.g., every 5-10 minutes) to process
new data from training clients.

The ingestion process supports two modes:
1. AnalyticsConfig mode (recommended): Load configuration from env.py
2. CLI args mode: Specify all settings via command line arguments

AnalyticsConfig mode automatically handles:
- Staging backend selection (S3 or filesystem)
- Central database configuration (DuckDB or DuckLake)
- S3 credentials and connection settings
"""
import argparse
import time
from pathlib import Path
from typing import Optional
import duckdb
from datetime import datetime

# Table configuration
MODEL_PARAMETER = "model_parameter"
TRAIN_STEP_METRIC = "train_step_metric"
CHECKPOINT_SAMPLE_METRIC = "checkpoint_sample_metric"

SYNC_TABLES = [MODEL_PARAMETER, TRAIN_STEP_METRIC, CHECKPOINT_SAMPLE_METRIC]
TYPES = ["int", "float", "text"]


def get_s3_client(s3_key: str, s3_secret: str, s3_endpoint: str):
    """Create S3 client for file operations"""
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id=s3_key,
        aws_secret_access_key=s3_secret,
        endpoint_url=s3_endpoint,
    )


def ensure_s3_credentials(conn, s3_key: str, s3_secret: str, s3_region: str, s3_endpoint: str):
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


def ensure_ingestion_state_table(conn):
    """Ensure the ingestion state tracking table exists"""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_state (
            file_path TEXT PRIMARY KEY,
            ingested_at TIMESTAMP,
            row_count INTEGER
        )
        """
    )


def get_processed_files(conn) -> set[str]:
    """Get set of already processed file paths"""
    result = conn.execute(
        "SELECT file_path FROM ingestion_state"
    ).fetchall()
    return {row[0] for row in result}


def mark_file_processed(conn, file_path: str, row_count: int):
    """Mark a file as processed"""
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


def ingest_table(
    conn,
    table_name: str,
    parquet_files: list[str],
    processed_files: set[str],
    dry_run: bool = False,
) -> int:
    """
    Ingest a single unified table, splitting by type

    Args:
        conn: DuckDB connection
        table_name: Name of the table to ingest
        parquet_files: List of parquet file paths (S3 or filesystem)
        processed_files: Set of already processed file paths
        dry_run: If True, don't actually modify the database

    Returns the number of files processed
    """
    files_to_process = [f for f in parquet_files if f not in processed_files]

    if not files_to_process:
        print(f"[ingest] No new files to process for {table_name}")
        return 0

    print(f"[ingest] Processing {len(files_to_process)} files for {table_name}")

    total_rows = 0

    for file_path in files_to_process:
        print(f"[ingest]   Processing {file_path}")

        if dry_run:
            print(f"[ingest]   DRY RUN - would process {file_path}")
            continue

        # Get row count
        row_count_result = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{file_path}')"
        ).fetchone()
        file_row_count = row_count_result[0] if row_count_result else 0

        # Split by type and insert into type-specific tables
        for type_name in TYPES:
            value_col = f"value_{type_name}"
            target_table = f"{table_name}_{type_name}"

            # Build column list based on table type
            if table_name == MODEL_PARAMETER:
                columns = "model_id, run_id, timestamp, name"
            elif table_name == TRAIN_STEP_METRIC:
                columns = "model_id, run_id, timestamp, name, step"
            elif table_name == CHECKPOINT_SAMPLE_METRIC:
                # Checkpoint sample metrics have arrays
                mean_col = f"mean_{type_name}"
                value_per_sample_col = f"value_per_sample_{type_name}"

                conn.execute(
                    f"""
                    INSERT INTO {target_table}
                    SELECT model_id, timestamp, step, name, dataset, sample_ids,
                           {mean_col} as mean,
                           {value_per_sample_col} as value_per_sample
                    FROM read_parquet('{file_path}')
                    WHERE type = '{type_name}' AND {mean_col} IS NOT NULL
                    """
                )
                continue

            # For model_parameter and train_step_metric
            conn.execute(
                f"""
                INSERT INTO {target_table}
                SELECT {columns}, {value_col} as value
                FROM read_parquet('{file_path}')
                WHERE type = '{type_name}' AND {value_col} IS NOT NULL
                """
            )

        # Mark file as processed
        mark_file_processed(conn, file_path, file_row_count)
        total_rows += file_row_count

    print(f"[ingest] Ingested {total_rows} total rows from {len(files_to_process)} files")
    return len(files_to_process)


def archive_processed_files(
    s3_client,
    bucket: str,
    staging_prefix: str,
    archive_prefix: str,
    processed_files: set[str],
):
    """Move processed files from staging to archive"""
    for s3_path in processed_files:
        if not s3_path.startswith(f"s3://{bucket}/{staging_prefix}"):
            continue

        # Extract the key
        key = s3_path.replace(f"s3://{bucket}/", "")
        filename = Path(key).name

        # Determine archive path
        table_name = key.split("/")[1]  # Extract table name from path
        archive_key = f"{archive_prefix}/{table_name}/{filename}"

        print(f"[archive] Moving {key} -> {archive_key}")
        move_s3_file(s3_client, bucket, key, archive_key)


def attach_ducklake(conn, postgres_host: str, postgres_port: int, postgres_password: str, s3_key: str, s3_secret: str, s3_region: str, s3_endpoint: str):
    """Attach to remote DuckLake database"""
    import os

    # Check for local ducklake mode
    if "DUCKLAKE" in os.environ:
        lake_name = os.environ["DUCKLAKE"]
        conn.sql(
            f"ATTACH IF NOT EXISTS 'ducklake:{lake_name}' as central (data_path './{lake_name}_data')"
        )
        print(f"[ingest] Attached to local ducklake at {lake_name}")
        return

    # Remote ducklake mode
    conn.execute("INSTALL ducklake")
    conn.execute("INSTALL aws")

    # Configure S3 for ducklake data path
    conn.execute(
        f"""CREATE OR REPLACE SECRET ducklake_s3 (
             TYPE s3,
             PROVIDER config,
             KEY_ID '{s3_key}',
             SECRET '{s3_secret}',
             REGION '{s3_region}',
             ENDPOINT '{s3_endpoint}'
         )
        """
    )

    conn.execute(
        f"ATTACH IF NOT EXISTS 'ducklake:postgres:user=postgres password={postgres_password} host={postgres_host} port={postgres_port} dbname=ducklake' as central (data_path 's3://eqpducklake')"
    )
    print(f"[ingest] Attached to remote DuckLake at {postgres_host}:{postgres_port}")


def ingest_all_from_config(config, dry_run: bool = False):
    """
    Ingest data using AnalyticsConfig

    This is the recommended entry point for ingestion. It automatically:
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
        print(f"[ingest] Using DuckLake central database")
        conn = duckdb.connect()  # In-memory connection

        # Attach DuckLake
        postgres = config.central.postgres
        s3 = config.central.s3

        conn.execute("INSTALL ducklake")
        conn.execute("INSTALL aws")

        # Configure S3 for DuckLake data path
        ensure_s3_credentials(conn, s3.key, s3.secret, s3.region, s3.endpoint)

        conn.execute(
            f"ATTACH IF NOT EXISTS 'ducklake:postgres:user={postgres.user} password={postgres.password} "
            f"host={postgres.host} port={postgres.port} dbname={postgres.dbname}' as central "
            f"(data_path '{config.central.data_path}')"
        )
        conn.execute("USE central")

    elif config.is_duckdb_central():
        print(f"[ingest] Using local DuckDB database: {config.central.db_path}")
        conn = duckdb.connect(str(config.central.db_path))
    else:
        raise ValueError(f"Unknown central database type: {config.central.type}")

    # Ensure ingestion state table
    ensure_ingestion_state_table(conn)

    # Get already processed files
    processed_files = get_processed_files(conn)
    print(f"[ingest] Already processed {len(processed_files)} files")

    total_files = 0

    # Dispatch based on staging type
    if config.is_s3_staging():
        print(f"[ingest] Using S3 staging: s3://{config.staging.bucket}/{config.staging.prefix}")

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

            # Archive processed files
            if not dry_run and files_processed > 0:
                newly_processed = set(s3_files) - processed_files
                archive_processed_files(
                    s3_client,
                    config.staging.bucket,
                    config.staging.prefix,
                    config.staging.archive_prefix,
                    newly_processed
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

            # Archive processed files
            if not dry_run and files_processed > 0:
                newly_processed = set(fs_files) - processed_files
                for file_path_str in newly_processed:
                    file_path = Path(file_path_str)
                    move_filesystem_file(file_path, archive_dir, table_name)

    else:
        raise ValueError(f"Unknown staging type: {config.staging.type}")

    print(f"[ingest] Ingestion complete. Processed {total_files} files.")
    conn.close()


def ingest_all(
    s3_bucket: str,
    s3_key: str,
    s3_secret: str,
    s3_region: str,
    s3_endpoint: str,
    db_path: str = None,
    use_ducklake: bool = False,
    postgres_host: str = None,
    postgres_port: int = None,
    postgres_password: str = None,
    staging_prefix: str = "staging",
    archive_prefix: str = "archive",
    dry_run: bool = False,
):
    """
    Main ingestion process

    Args:
        s3_bucket: S3 bucket name
        s3_key: S3 access key
        s3_secret: S3 secret key
        s3_region: S3 region
        s3_endpoint: S3 endpoint URL
        db_path: Path to central DuckDB database (local mode)
        use_ducklake: Use DuckLake instead of local file
        postgres_host: Postgres host for DuckLake
        postgres_port: Postgres port for DuckLake
        postgres_password: Postgres password for DuckLake
        staging_prefix: Prefix for staging files
        archive_prefix: Prefix for archived files
        dry_run: If True, don't actually modify the database
    """
    print(f"[ingest] Starting ingestion from s3://{s3_bucket}/{staging_prefix}")

    # Connect to central database
    if use_ducklake:
        print(f"[ingest] Using DuckLake mode")
        conn = duckdb.connect()  # In-memory connection
        attach_ducklake(conn, postgres_host, postgres_port, postgres_password, s3_key, s3_secret, s3_region, s3_endpoint)
        conn.execute("USE central")
    else:
        print(f"[ingest] Using local database: {db_path}")
        conn = duckdb.connect(db_path)

    print(f"[ingest] Dry run: {dry_run}")

    # Configure S3
    ensure_s3_credentials(conn, s3_key, s3_secret, s3_region, s3_endpoint)

    # Ensure ingestion state table
    ensure_ingestion_state_table(conn)

    # Get already processed files
    processed_files = get_processed_files(conn)
    print(f"[ingest] Already processed {len(processed_files)} files")

    # Get S3 client for file operations
    s3_client = get_s3_client(s3_key, s3_secret, s3_endpoint)

    # Process each table
    total_files = 0
    for table_name in SYNC_TABLES:
        table_prefix = f"{staging_prefix}/{table_name}"

        # List files in S3
        s3_files = list_s3_files(s3_client, s3_bucket, table_prefix)
        print(f"[ingest] Found {len(s3_files)} files for {table_name}")

        # Ingest
        files_processed = ingest_table(
            conn, table_name, s3_files, processed_files, dry_run
        )
        total_files += files_processed

        # Archive processed files
        if not dry_run and files_processed > 0:
            newly_processed = set(s3_files) - processed_files
            archive_processed_files(
                s3_client, s3_bucket, staging_prefix, archive_prefix, newly_processed
            )

    print(f"[ingest] Ingestion complete. Processed {total_files} files.")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest unified Parquet files from staging into central database"
    )

    # Config mode
    parser.add_argument(
        "--config",
        action="store_true",
        help="Use AnalyticsConfig from env.py (recommended). Overrides all other settings.",
    )

    # Database mode selection (for legacy CLI args mode)
    db_group = parser.add_mutually_exclusive_group()
    db_group.add_argument(
        "--db", help="Path to local DuckDB database file (legacy mode)"
    )
    db_group.add_argument(
        "--ducklake", action="store_true", help="Use DuckLake (remote) instead of local file (legacy mode)"
    )

    # Postgres settings (for DuckLake)
    parser.add_argument("--postgres-host", help="Postgres host for DuckLake (defaults to ComputeEnvironment.postgres_host)")
    parser.add_argument("--postgres-port", type=int, help="Postgres port for DuckLake (defaults to ComputeEnvironment.postgres_port)")
    parser.add_argument("--postgres-password", help="Postgres password for DuckLake (defaults to ComputeEnvironment.postgres_password)")

    # S3 settings
    parser.add_argument("--s3-bucket", help="S3 bucket name (defaults to ComputeEnvironment.s3_bucket)")
    parser.add_argument("--s3-key", help="S3 access key (defaults to ComputeEnvironment.s3_key)")
    parser.add_argument("--s3-secret", help="S3 secret key (defaults to ComputeEnvironment.s3_secret)")
    parser.add_argument("--s3-region", help="S3 region (defaults to ComputeEnvironment.s3_region or us-east-1)")
    parser.add_argument("--s3-endpoint", help="S3 endpoint URL (defaults to ComputeEnvironment.s3_endpoint)")
    parser.add_argument(
        "--staging-prefix", default="staging", help="S3 prefix for staging files"
    )
    parser.add_argument(
        "--archive-prefix", default="archive", help="S3 prefix for archived files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't modify database"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously (polling mode)",
    )
    parser.add_argument(
        "--interval", type=int, default=300, help="Polling interval in seconds"
    )

    args = parser.parse_args()

    # ============================================================================
    # Config mode - Use AnalyticsConfig from env.py
    # ============================================================================
    if args.config:
        from lib.analytics_config import analytics_config

        config = analytics_config()

        print(f"[ingest] Using AnalyticsConfig from env.py")
        print(f"[ingest]   Staging: {config.staging.type}")
        print(f"[ingest]   Central: {config.central.type}")

        if args.continuous:
            print(f"[ingest] Running in continuous mode (interval: {args.interval}s)")
            while True:
                try:
                    ingest_all_from_config(config, dry_run=args.dry_run)
                except Exception as e:
                    print(f"[ingest] Error during ingestion: {e}")
                    import traceback
                    traceback.print_exc()

                print(f"[ingest] Sleeping for {args.interval} seconds...")
                time.sleep(args.interval)
        else:
            ingest_all_from_config(config, dry_run=args.dry_run)

        return

    # ============================================================================
    # Legacy CLI args mode
    # ============================================================================
    print("[ingest] WARNING: Using legacy CLI args mode. Consider using --config instead.")

    # Load defaults from ComputeEnvironment if not provided via CLI
    from lib.compute_env import env
    e = env()

    # Determine database mode
    use_ducklake = args.ducklake
    db_path = args.db

    # If neither specified, try to use ComputeEnvironment defaults
    if not use_ducklake and not db_path:
        # Default to DuckLake if postgres settings are available
        if e.postgres_host:
            use_ducklake = True
            print("[ingest] No --db or --ducklake specified, using DuckLake from ComputeEnvironment")
        else:
            parser.error("Must specify either --db (local file) or --ducklake (or configure ComputeEnvironment)")

    # S3 settings (with ComputeEnvironment fallback)
    s3_bucket = args.s3_bucket or e.s3_bucket
    s3_key = args.s3_key or e.s3_key
    s3_secret = args.s3_secret or e.s3_secret
    s3_region = args.s3_region or e.s3_region or "us-east-1"
    s3_endpoint = args.s3_endpoint or e.s3_endpoint

    # Postgres settings (for DuckLake)
    postgres_host = args.postgres_host or e.postgres_host
    postgres_port = args.postgres_port or e.postgres_port
    postgres_password = args.postgres_password or e.postgres_password

    # Validate required settings
    if not s3_bucket:
        parser.error("--s3-bucket is required (or set in ComputeEnvironment)")
    if not s3_key:
        parser.error("--s3-key is required (or set in ComputeEnvironment)")
    if not s3_secret:
        parser.error("--s3-secret is required (or set in ComputeEnvironment)")
    if not s3_endpoint:
        parser.error("--s3-endpoint is required (or set in ComputeEnvironment)")

    if use_ducklake:
        if not postgres_host or not postgres_port or not postgres_password:
            parser.error("DuckLake mode requires postgres settings (or set in ComputeEnvironment)")

    print(f"[ingest] Configuration:")
    print(f"[ingest]   Database: {'DuckLake' if use_ducklake else db_path}")
    if use_ducklake:
        print(f"[ingest]   Postgres: {postgres_host}:{postgres_port}")
    print(f"[ingest]   S3 Bucket: {s3_bucket}")
    print(f"[ingest]   S3 Region: {s3_region}")
    print(f"[ingest]   S3 Endpoint: {s3_endpoint}")

    if args.continuous:
        print(f"[ingest] Running in continuous mode (interval: {args.interval}s)")
        while True:
            try:
                ingest_all(
                    s3_bucket=s3_bucket,
                    s3_key=s3_key,
                    s3_secret=s3_secret,
                    s3_region=s3_region,
                    s3_endpoint=s3_endpoint,
                    db_path=db_path,
                    use_ducklake=use_ducklake,
                    postgres_host=postgres_host,
                    postgres_port=postgres_port,
                    postgres_password=postgres_password,
                    staging_prefix=args.staging_prefix,
                    archive_prefix=args.archive_prefix,
                    dry_run=args.dry_run,
                )
            except Exception as e:
                print(f"[ingest] Error during ingestion: {e}")
                import traceback
                traceback.print_exc()

            print(f"[ingest] Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        ingest_all(
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            s3_secret=s3_secret,
            s3_region=s3_region,
            s3_endpoint=s3_endpoint,
            db_path=db_path,
            use_ducklake=use_ducklake,
            postgres_host=postgres_host,
            postgres_port=postgres_port,
            postgres_password=postgres_password,
            staging_prefix=args.staging_prefix,
            archive_prefix=args.archive_prefix,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
