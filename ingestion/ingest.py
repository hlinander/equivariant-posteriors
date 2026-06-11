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
from datetime import datetime, timezone
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
# Larger batches amortize fixed per-batch overhead (temp-table build, type-split
# INSERTs, per-file count, DuckLake snapshot) now that reads are cheap, and keep the
# read threads saturated. RAM is ample (200 small files ~2MB; 200 checkpoint ~200MB).
INGEST_BATCH_SIZE = 200
# Staging reads from S3 are latency-bound on small files (~0.4s round-trip each);
# a probe showed ~3x more files/s at 16-way vs the 8-core default. RAM is ample, so
# oversubscribe threads to keep more concurrent GETs in flight.
S3_READ_THREADS = 16
# Archiving (staging->archive copy + delete) is server-side; under the storage
# throttle it's backend-bound, so concurrency has a low ceiling (a probe showed
# ~4/6/8 files/s at 10/32/64 workers). Still worth running wide. Default boto3 pool
# is 10, so the client pool must be at least this large or workers queue on sockets.
ARCHIVE_COPY_WORKERS = 64
S3_CLIENT_POOL = 80
_SCHEMA_ENSURED = False

# Observability: a single append-only event log is the source of truth; the legacy
# names below are recreated as SQL views over it for dashboard compatibility.
INGEST_EVENTS_TABLE = "ingest_events"
INGEST_METRICS_TABLE = "ingest_metrics"
INGEST_PROGRESS_TABLE = "ingest_progress"
INGEST_STAGING_SAMPLE_TABLE = "ingest_staging_sample"


def get_s3_client(s3_key: str, s3_secret: str, s3_endpoint: str, s3_url_style: str = "path"):
    """Create S3 client for file operations"""
    import boto3
    from botocore.config import Config

    # Ensure endpoint has a protocol (boto3 requires it)
    if not s3_endpoint.startswith("http://") and not s3_endpoint.startswith("https://"):
        # Default to https if no protocol specified
        s3_endpoint = f"https://{s3_endpoint}"

    # Bucket addressing: "path" (endpoint/bucket/key — MinIO, Hetzner) or
    # "vhost" (bucket.endpoint/key — virtual-hosted, e.g. Hexabyte). boto3 calls
    # the virtual-hosted style "virtual".
    addressing = "virtual" if s3_url_style == "vhost" else "path"
    return boto3.client(
        "s3",
        aws_access_key_id=s3_key,
        aws_secret_access_key=s3_secret,
        endpoint_url=s3_endpoint,
        config=Config(
            s3={"addressing_style": addressing},
            max_pool_connections=S3_CLIENT_POOL,
        ),
    )


def ensure_s3_credentials(
    conn,
    s3_key: str,
    s3_secret: str,
    s3_region: str,
    s3_endpoint: str,
    s3_url_style: str = "path",
):
    """Ensure S3 credentials are configured in DuckDB"""
    conn.execute("INSTALL aws")
    conn.execute("LOAD aws")

    # path-style: endpoint/bucket/key (MinIO default)
    # vhost-style: bucket.endpoint/key (AWS default, some providers require it)
    conn.execute(f"SET s3_url_style='{s3_url_style}'")

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
            URL_STYLE '{s3_url_style}',
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


def ensure_events_table(conn):
    """Ensure the single append-only ingestion event log exists (idempotent).

    ingest_events is the one source of truth for ingestion analytics; everything
    else (staging depth, ingest/archive progress, per-cycle metrics, the timeline)
    derives from it via SQL. Columns are generic; their meaning depends on
    event_type (see record_events callers and ensure_compat_views):

        cycle_started_at  groups all events of one cycle
        event_type        cycle_start | staging_sample | table_start | ingest_batch
                          | ingest_done | archive_start | archive_chunk | archive_done
                          | table_error
        table_name        NULL for cycle-level events
        seq               batch / chunk index
        file_count        files this event concerns (staged / batch / ingested / archived)
        byte_count        bytes (staging_sample)
        row_count         rows (ingest_done)
        total_files       the phase's target (denominator for a burn-up)
        detail            error message / free text
        files             optional in-flight file list (ingest_batch)
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {INGEST_EVENTS_TABLE} (
            ts               TIMESTAMPTZ,
            cycle_started_at TIMESTAMPTZ,
            event_type       TEXT,
            table_name       TEXT,
            seq              INTEGER,
            file_count       BIGINT,
            byte_count       BIGINT,
            row_count        BIGINT,
            total_files      BIGINT,
            detail           TEXT,
            files            TEXT[]
        )
        """
    )


def ensure_compat_views(conn):
    """Recreate the legacy table names as views over ingest_events so existing
    dashboards keep working. Best-effort — never break ingestion if a view fails."""
    views = {
        INGEST_STAGING_SAMPLE_TABLE: f"""
            SELECT cycle_started_at, table_name,
                   file_count AS staged_files, byte_count AS staged_bytes, ts AS sampled_at
            FROM {INGEST_EVENTS_TABLE} WHERE event_type = 'staging_sample'
        """,
        INGEST_PROGRESS_TABLE: f"""
            SELECT cycle_started_at, table_name,
                   seq AS batch_index, file_count AS batch_file_count,
                   files AS batch_files, total_files AS table_total_files, ts AS started_at
            FROM {INGEST_EVENTS_TABLE} WHERE event_type = 'ingest_batch'
        """,
        INGEST_METRICS_TABLE: f"""
            SELECT cycle_started_at, table_name,
                   max(file_count) FILTER (WHERE event_type='staging_sample') AS staging_file_count,
                   max(byte_count) FILTER (WHERE event_type='staging_sample') AS staging_bytes,
                   max(file_count) FILTER (WHERE event_type='ingest_done')    AS files_ingested,
                   max(row_count)  FILTER (WHERE event_type='ingest_done')    AS rows_ingested,
                   max(file_count) FILTER (WHERE event_type='archive_done')   AS files_archived,
                   date_diff('second',
                             min(ts) FILTER (WHERE event_type='table_start'),
                             max(ts) FILTER (WHERE event_type IN ('archive_done','ingest_done','table_error')))
                     AS duration_seconds,
                   any_value(detail) FILTER (WHERE event_type='table_error') AS error,
                   max(ts) AS recorded_at
            FROM {INGEST_EVENTS_TABLE}
            WHERE table_name IS NOT NULL
            GROUP BY cycle_started_at, table_name
        """,
    }
    for name, select_sql in views.items():
        # Old schema stored these as base tables; drop so a view can take the name.
        # DROP TABLE on an existing view errors ("not a table") — that's fine, it just
        # means the migration already happened; swallow it.
        try:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        except Exception:
            pass
        try:
            conn.execute(f"CREATE OR REPLACE VIEW {name} AS {select_sql}")
        except Exception as e:
            print(f"[ingest] WARNING: could not create compat view {name}: {e}")


def record_events(conn, events: list[dict]):
    """Append rows to the ingest_events log. Each event dict has 'event_type' plus
    any of: cycle_started_at, table_name, seq, file_count, byte_count, row_count,
    total_files, detail, files (list[str]). ts is stamped server-side via now()."""
    if not events:
        return
    values_sql = []
    params = []
    for e in events:
        files = e.get("files")
        if files:
            files_lit = "[" + ", ".join("'" + f.replace("'", "''") + "'" for f in files) + "]"
        else:
            files_lit = "NULL"
        values_sql.append(f"(now(), ?, ?, ?, ?, ?, ?, ?, ?, ?, {files_lit})")
        params.extend(
            [
                e.get("cycle_started_at"),
                e["event_type"],
                e.get("table_name"),
                e.get("seq"),
                e.get("file_count"),
                e.get("byte_count"),
                e.get("row_count"),
                e.get("total_files"),
                e.get("detail"),
            ]
        )
    conn.execute(
        f"""
        INSERT INTO {INGEST_EVENTS_TABLE}
            (ts, cycle_started_at, event_type, table_name, seq,
             file_count, byte_count, row_count, total_files, detail, files)
        VALUES {", ".join(values_sql)}
        """,
        params,
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


def list_s3_files_with_sizes(s3_client, bucket: str, prefix: str) -> list[tuple[str, int]]:
    """List parquet files in an S3 prefix with their sizes, in a single listing pass."""
    out = []
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                out.append((f"s3://{bucket}/{key}", obj["Size"]))

    return out


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

    # Read the batch from S3 exactly ONCE into a local temp table, then run every
    # type-split insert and the per-file count against local memory. Previously each
    # file was re-fetched from (throttled) S3 2-4x: 3 type INSERTs + 1 COUNT for the
    # split tables. RAM is ample, so this is a pure win on the bottleneck.
    src = "_batch"
    conn.execute(f"CREATE OR REPLACE TEMP TABLE {src} AS SELECT * FROM {pq}")

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
                    FROM {src}
                    WHERE type = '{type_name}' AND {value_col} IS NOT NULL
                    """
                )
            elif table_name == TRAIN_STEP_METRIC:
                conn.execute(
                    f"""
                    INSERT INTO {target_table} BY NAME
                    SELECT model_id, run_id, timestamp, name, step,
                           {value_col} as value
                    FROM {src}
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
                    FROM {src}
                    WHERE type = '{type_name}' AND {mean_col} IS NOT NULL
                    """
                )
    else:
        conn.execute(
            f"""
            INSERT INTO {table_name} BY NAME
            SELECT * EXCLUDE (filename) FROM {src}
            """
        )

    # Per-file row counts from the local copy (no extra S3 read)
    per_file_counts = dict(
        conn.execute(f"SELECT filename, COUNT(*) FROM {src} GROUP BY filename").fetchall()
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
    cycle_started_at=None,
    write_progress: bool = False,
) -> tuple[int, int]:
    """
    Ingest a single unified table, splitting by type.

    Files are processed in batches to balance S3 round-trips against memory.

    Returns (files_processed, rows_ingested).
    """
    files_to_process = [f for f in parquet_files if f not in processed_files]

    if not files_to_process:
        print(f"[ingest] No new files to process for {table_name}")
        return 0, 0

    print(f"[ingest] Processing {len(files_to_process)} files for {table_name}")

    if dry_run:
        for f in files_to_process:
            print(f"[ingest]   DRY RUN - would process {f}")
        return 0, 0

    emit = write_progress and cycle_started_at is not None
    total_rows = 0
    total_to_process = len(files_to_process)
    if emit:
        record_events(conn, [{
            "event_type": "table_start", "cycle_started_at": cycle_started_at,
            "table_name": table_name, "total_files": total_to_process,
        }])
    for i in range(0, total_to_process, batch_size):
        batch = files_to_process[i : i + batch_size]
        batch_index = i // batch_size + 1
        print(f"[ingest]   Batch {batch_index}: {len(batch)} files")
        # Heartbeat the batch before reading it, so a dashboard sees in-flight work
        # even while a slow batch is still being fetched from S3.
        if emit:
            record_events(conn, [{
                "event_type": "ingest_batch", "cycle_started_at": cycle_started_at,
                "table_name": table_name, "seq": batch_index, "file_count": len(batch),
                "total_files": total_to_process, "files": batch,
            }])
        total_rows += _ingest_batch(conn, table_name, batch)

    if emit:
        record_events(conn, [{
            "event_type": "ingest_done", "cycle_started_at": cycle_started_at,
            "table_name": table_name, "file_count": total_to_process, "row_count": total_rows,
        }])
    print(
        f"[ingest] Ingested {total_rows} total rows from {len(files_to_process)} files"
    )
    return total_to_process, total_rows


def archive_processed_files(
    s3_client,
    bucket: str,
    staging_prefix: str,
    archive_prefix: str,
    processed_files: set[str],
    conn=None,
    cycle_started_at=None,
    table_name=None,
    write_events: bool = False,
) -> int:
    """Move processed files from staging to archive. Returns the number archived.

    Copies are parallelized with threads, then originals are batch-deleted. Emits
    archive_start / archive_chunk / archive_done events when write_events is set.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Build list of (source_key, dest_key) pairs
    moves = []
    for s3_path in processed_files:
        if not s3_path.startswith(f"s3://{bucket}/{staging_prefix}"):
            continue

        key = s3_path.replace(f"s3://{bucket}/", "")
        filename = Path(key).name
        key_table = key.split("/")[1]
        archive_key = f"{archive_prefix}/{key_table}/{filename}"
        moves.append((key, archive_key))

    if not moves:
        return 0

    emit = write_events and conn is not None and cycle_started_at is not None
    if emit:
        record_events(conn, [{
            "event_type": "archive_start", "cycle_started_at": cycle_started_at,
            "table_name": table_name, "total_files": len(moves),
        }])

    print(f"[archive] Archiving {len(moves)} files")

    # Parallel copy. Tolerate a missing source: under heavy delete load RGW listing
    # can briefly return an already-deleted key, and one such file must not abort the
    # whole table's archive (which would keep staging from ever draining).
    from botocore.exceptions import ClientError

    def copy_one(src_dest):
        src, dest = src_dest
        try:
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": src},
                Key=dest,
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NoSuchObject"):
                return  # source already gone — effectively already archived
            raise

    # Move in durable chunks: parallel-copy a chunk, then delete those sources,
    # before moving to the next. A restart (e.g. a deploy) loses at most one chunk
    # of progress instead of the whole archive — the previous copy-all-then-delete
    # -all meant any interruption re-copied every file next cycle (straggler pile-up).
    CHUNK = 1000
    done = 0
    with ThreadPoolExecutor(max_workers=ARCHIVE_COPY_WORKERS) as pool:
        for i in range(0, len(moves), CHUNK):
            chunk = moves[i : i + CHUNK]
            list(pool.map(copy_one, chunk))  # raises on first error
            s3_client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": src} for src, _ in chunk]},
            )
            done += len(chunk)
            if emit:
                record_events(conn, [{
                    "event_type": "archive_chunk", "cycle_started_at": cycle_started_at,
                    "table_name": table_name, "seq": i // CHUNK + 1,
                    "file_count": len(chunk), "total_files": len(moves),
                }])
            print(f"[archive]   Moved {done}/{len(moves)} files")

    if emit:
        record_events(conn, [{
            "event_type": "archive_done", "cycle_started_at": cycle_started_at,
            "table_name": table_name, "file_count": len(moves),
        }])
    print(f"[archive] Archived {len(moves)} files")
    return len(moves)


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
        ensure_s3_credentials(conn, s3.key, s3.secret, s3.region, s3.endpoint, s3.url_style)

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

    # Oversubscribe threads: staging reads are S3-latency-bound, so more concurrent
    # GETs beat the per-core default (see S3_READ_THREADS).
    conn.execute(f"SET threads = {S3_READ_THREADS}")

    # Ensure schema exists (once per process, not every ingestion cycle)
    global _SCHEMA_ENSURED
    if not _SCHEMA_ENSURED:
        print("[ingest] Ensuring central database schema exists")
        ensure_central_schema(conn)
        ensure_ingestion_state_table(conn)
        ensure_events_table(conn)
        ensure_compat_views(conn)
        _SCHEMA_ENSURED = True

    # Get already processed files
    print("[ingest] Getting processed files...")
    processed_files = get_processed_files(conn)
    print(f"[ingest] Already processed {len(processed_files)} files")

    total_files = 0
    # Per-cycle observability. cycle_started_at ties together every event of this
    # pass in the ingest_events log; all dashboards/metrics derive from that log.
    # errors collects per-table failures so one bad table (e.g. a checkpoint timeout)
    # no longer aborts the cycle — it's recorded as a table_error event and we go on.
    cycle_started_at = datetime.now(timezone.utc)
    errors: list[tuple[str, Exception]] = []

    def _record_events(events):
        # Event writes must never break ingestion — log and continue on failure.
        if dry_run or not events:
            return
        try:
            record_events(conn, events)
        except Exception as e:
            print(f"[ingest] WARNING: failed to record events: {e}")

    try:
        # Dispatch based on staging type
        if config.is_s3_staging():
            print(
                f"[ingest] Using S3 staging: s3://{config.staging.bucket}/{config.staging.prefix}"
            )

            s3 = config.staging.s3

            # Configure S3 credentials in DuckDB (for read_parquet)
            ensure_s3_credentials(conn, s3.key, s3.secret, s3.region, s3.endpoint, s3.url_style)

            # Get S3 client for file operations (list/move)
            s3_client = get_s3_client(s3.key, s3.secret, s3.endpoint, s3.url_style)

            # Top-of-cycle staging snapshot + list cache. List every table once
            # here, reuse the result for ingestion below (no second listing), and
            # record the depth so a dashboard knows each table's target up front
            # (independent of the sequential drain) and sees arrivals sampled
            # pre-drain. A listing failure is logged and that table falls back to a
            # fresh list in the loop. Never let sampling break ingestion.
            #
            # Listing is done concurrently across tables: each table's pagination is
            # inherently sequential (continuation tokens), so the wall time is the
            # slowest single table rather than the sum over all of them.
            from concurrent.futures import ThreadPoolExecutor

            def _list_table(table_name):
                try:
                    s = list_s3_files_with_sizes(
                        s3_client, config.staging.bucket,
                        f"{config.staging.prefix}/{table_name}",
                    )
                    return table_name, s, None
                except Exception as e:
                    return table_name, None, e

            staged_cache: dict[str, list] = {}
            samples = []
            with ThreadPoolExecutor(max_workers=len(SYNC_TABLES)) as ex:
                for table_name, s, e in ex.map(_list_table, SYNC_TABLES):
                    if e is not None:
                        print(f"[ingest] WARNING: staging sample/list failed for {table_name}: {e}")
                        continue
                    staged_cache[table_name] = s
                    samples.append((table_name, len(s), sum(sz for _, sz in s)))
            # cycle_start + a staging_sample event per table (every table's target up front)
            _record_events(
                [{"event_type": "cycle_start", "cycle_started_at": cycle_started_at}]
                + [{"event_type": "staging_sample", "cycle_started_at": cycle_started_at,
                    "table_name": t, "file_count": n, "byte_count": b} for (t, n, b) in samples]
            )

            # Process each table
            for table_name in SYNC_TABLES:
                table_prefix = f"{config.staging.prefix}/{table_name}"
                try:
                    # Reuse the cycle-start listing; fall back to a fresh list only
                    # if the up-front list failed for this table.
                    if table_name in staged_cache:
                        staged = staged_cache[table_name]
                    else:
                        staged = list_s3_files_with_sizes(
                            s3_client, config.staging.bucket, table_prefix
                        )
                    s3_files = [p for p, _ in staged]
                    print(f"[ingest] Found {len(s3_files)} files for {table_name}")

                    # Ingest
                    files_processed, rows_ingested = ingest_table(
                        conn, table_name, s3_files, processed_files, dry_run,
                        cycle_started_at=cycle_started_at, write_progress=not dry_run,
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
                            conn=conn, cycle_started_at=cycle_started_at,
                            table_name=table_name, write_events=True,
                        )
                except Exception as e:
                    errors.append((table_name, e))
                    print(f"[ingest] ERROR processing {table_name}: {e}")
                    import traceback

                    traceback.print_exc()
                    _record_events([{
                        "event_type": "table_error", "cycle_started_at": cycle_started_at,
                        "table_name": table_name, "detail": str(e),
                    }])

        elif config.is_filesystem_staging():
            print(f"[ingest] Using filesystem staging: {config.staging.staging_dir}")

            staging_dir = Path(config.staging.staging_dir)
            archive_dir = Path(config.staging.archive_dir)

            # Top-of-cycle staging snapshot (see S3 branch for rationale).
            samples = []
            for table_name in SYNC_TABLES:
                try:
                    fs = list_filesystem_files(staging_dir, table_name)
                    samples.append(
                        (table_name, len(fs),
                         sum(Path(f).stat().st_size for f in fs if Path(f).exists()))
                    )
                except Exception as e:
                    print(f"[ingest] WARNING: staging sample failed for {table_name}: {e}")
            _record_events(
                [{"event_type": "cycle_start", "cycle_started_at": cycle_started_at}]
                + [{"event_type": "staging_sample", "cycle_started_at": cycle_started_at,
                    "table_name": t, "file_count": n, "byte_count": b} for (t, n, b) in samples]
            )

            # Process each table
            for table_name in SYNC_TABLES:
                try:
                    # List files in staging directory
                    fs_files = list_filesystem_files(staging_dir, table_name)
                    print(f"[ingest] Found {len(fs_files)} files for {table_name}")

                    # Ingest (DuckDB's read_parquet works with filesystem paths)
                    files_processed, rows_ingested = ingest_table(
                        conn, table_name, fs_files, processed_files, dry_run,
                        cycle_started_at=cycle_started_at, write_progress=not dry_run,
                    )
                    total_files += files_processed

                    # Archive all staging files that have been ingested
                    if not dry_run:
                        for file_path_str in fs_files:
                            file_path = Path(file_path_str)
                            move_filesystem_file(file_path, archive_dir, table_name)
                except Exception as e:
                    errors.append((table_name, e))
                    print(f"[ingest] ERROR processing {table_name}: {e}")
                    import traceback

                    traceback.print_exc()
                    _record_events([{
                        "event_type": "table_error", "cycle_started_at": cycle_started_at,
                        "table_name": table_name, "detail": str(e),
                    }])

        else:
            raise ValueError(f"Unknown staging type: {config.staging.type}")

        print(f"[ingest] Ingestion complete. Processed {total_files} files.")
    finally:
        conn.close()

    # Surface table failures to the caller (continuous mode pings the healthcheck
    # /fail), but only after every table has had its chance to run this cycle.
    if errors:
        summary = ", ".join(f"{t}: {e}" for t, e in errors)
        raise RuntimeError(f"Ingestion completed with {len(errors)} table error(s): {summary}")


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
