"""
Unified export module using AnalyticsConfig.

Dispatches to either S3 or filesystem staging based on configuration.
"""
from pathlib import Path
from typing import Optional
from lib.train_dataclasses import TrainRun
from lib.analytics_config import analytics_config, AnalyticsConfig
import lib.render_duck as duck


def export_all(train_run: TrainRun, config: Optional[AnalyticsConfig] = None) -> list:
    """
    Export all metric tables using the configured staging backend.

    Args:
        train_run: The training run context
        config: Analytics configuration (defaults to analytics_config())

    Returns:
        List of paths (S3 or filesystem) that were created
    """
    if config is None:
        config = analytics_config()

    # Get thread-safe cursor
    if duck.CONN is None:
        print("[export] Error: DuckDB connection not initialized")
        return []

    cursor = duck.CONN.cursor()

    # Dispatch based on staging type
    if config.is_s3_staging():
        from lib.export_parquet import flush_all_to_s3
        from lib.export_parquet import ensure_s3_credentials

        s3 = config.staging.s3
        ensure_s3_credentials(cursor)

        return flush_all_to_s3(
            train_run=train_run,
            s3_bucket=config.staging.bucket,
            cursor=cursor,
            s3_prefix=config.staging.prefix,
        )

    elif config.is_filesystem_staging():
        from lib.staging_filesystem import flush_all_to_filesystem

        return flush_all_to_filesystem(
            train_run=train_run,
            staging_dir=config.staging.staging_dir,
            cursor=cursor,
        )

    else:
        raise ValueError(f"Unknown staging type: {config.staging.type}")


def start_periodic_export(
    train_run: TrainRun,
    config: Optional[AnalyticsConfig] = None,
    interval_seconds: Optional[int] = None,
):
    """
    Start a background thread that periodically exports data.

    Args:
        train_run: The training run context
        config: Analytics configuration (defaults to analytics_config())
        interval_seconds: Override export interval from config

    Returns:
        The background thread handle
    """
    if config is None:
        config = analytics_config()

    if interval_seconds is None:
        interval_seconds = config.export_interval_seconds

    # Dispatch based on staging type
    if config.is_s3_staging():
        from lib.export_parquet import export_periodic

        return export_periodic(
            train_run=train_run,
            interval_seconds=interval_seconds,
            s3_bucket=config.staging.bucket,
        )

    elif config.is_filesystem_staging():
        from lib.staging_filesystem import export_periodic_filesystem

        return export_periodic_filesystem(
            train_run=train_run,
            staging_dir=config.staging.staging_dir,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(f"Unknown staging type: {config.staging.type}")
