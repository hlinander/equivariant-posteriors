"""
Example environment configuration.

Copy this to env.py and customize for your deployment.
On first run, env.py is auto-created with local filesystem defaults.
"""
from pathlib import Path
from lib.analytics_config import (
    AnalyticsConfig,
    S3Config,
    PostgresConfig,
    StagingS3,
    StagingFilesystem,
    CentralDuckDB,
    CentralDuckLake,
)
from lib.compute_env_config import ComputeEnvironment, Paths


# Base directory for all local data
LOCAL_DIR = Path(".local")


def get_env() -> ComputeEnvironment:
    """Configure paths for checkpoints, artifacts, etc."""
    return ComputeEnvironment(
        paths=Paths(
            checkpoints=LOCAL_DIR / "checkpoints",
            locks=LOCAL_DIR / "locks",
            distributed_requests=LOCAL_DIR / "distributed_requests",
            artifacts=LOCAL_DIR / "artifacts",
            datasets=LOCAL_DIR / "datasets",
        ),
    )


def get_analytics_config() -> AnalyticsConfig:
    """
    Configure the analytics pipeline.

    Flow: Training Client → Staging → Central Database
    """

    # -------------------------------------------------------------------------
    # Option 1: Local Filesystem (default)
    # -------------------------------------------------------------------------
    return AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=LOCAL_DIR / "staging",
            archive_dir=LOCAL_DIR / "archive",
        ),
        central=CentralDuckDB(
            db_path=LOCAL_DIR / "analytics.db",
        ),
    )

    # -------------------------------------------------------------------------
    # Option 2: S3 Staging + Local DB (for testing S3)
    # -------------------------------------------------------------------------
    # s3 = S3Config(
    #     key="your-access-key",
    #     secret="your-secret-key",
    #     region="us-east-1",
    #     endpoint="https://s3.amazonaws.com",
    # )
    # return AnalyticsConfig(
    #     staging=StagingS3(
    #         s3=s3,
    #         bucket="metrics-staging",
    #         prefix="staging",
    #         archive_prefix="archive",
    #     ),
    #     central=CentralDuckDB(
    #         db_path=LOCAL_DIR / "analytics.db",
    #     ),
    # )

    # -------------------------------------------------------------------------
    # Option 3: S3 Staging + DuckLake (production)
    # -------------------------------------------------------------------------
    # s3 = S3Config(
    #     key="your-access-key",
    #     secret="your-secret-key",
    #     region="us-east-1",
    #     endpoint="https://s3.amazonaws.com",
    # )
    # return AnalyticsConfig(
    #     staging=StagingS3(
    #         s3=s3,
    #         bucket="metrics-staging",
    #     ),
    #     central=CentralDuckLake(
    #         postgres=PostgresConfig(
    #             host="postgres.example.com",
    #             password="your-password",
    #         ),
    #         s3=s3,
    #         data_path="s3://ducklake-data",
    #     ),
    #     export_interval_seconds=300,
    #     ingest_interval_seconds=300,
    # )
