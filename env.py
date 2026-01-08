"""
Example environment configuration.

Copy this to env.py and customize for your deployment.
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


# ============================================================================
# Analytics Pipeline Configuration
# ============================================================================


def get_analytics_config() -> AnalyticsConfig:
    """
    Configure the analytics pipeline.

    This defines the entire flow:
      Training Client → Staging → Central Database

    Uncomment the configuration that matches your deployment.
    """

    # ---------------------------------------------------------------------------
    # Option 3: Production HPC (S3 + DuckLake)
    # ---------------------------------------------------------------------------
    # Perfect for: HPC cluster with shared central database
    # Note: Same S3 config used for both staging and DuckLake storage
    s3_prod = S3Config(
        key="2SDAPICIORG1Q27GVODS",
        secret="ChqPOQHROWJ4KbnXRbZRvOYA1azW9ji2nvbaxew4",
        region="hel1",
        endpoint="hel1.your-objectstorage.com",
    )
    postgres_prod = PostgresConfig(
        host="37.27.45.22",
        port=5430,
        user="postgres",
        password="uwvMtRDZsPMT5u",
        dbname="dl2",
    )
    return AnalyticsConfig(
        staging=StagingS3(
            s3=s3_prod,
            bucket="eqp-metrics-staging2",
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckLake(
            postgres=postgres_prod,
            s3=s3_prod,  # Same S3 config!
            data_path="s3://eqpducklake2",
        ),
        export_interval_seconds=300,  # Export every 5 minutes
        ingest_interval_seconds=20,
    )



def get_env() -> ComputeEnvironment:
    """
    Configure training client settings.

    This is separate from analytics config - it's for training-specific
    settings like checkpoint paths, dataset paths, etc.
    """
    return ComputeEnvironment(
        paths=Paths(
            checkpoints=Path("/proj/heal_pangu/eqp_climate/checkpoints"),
            artifacts=Path("/proj/heal_pangu/eqp_climate/artifacts"),
            datasets=Path("/proj/heal_pangu/datasets"), # points to dataset on cluster proj
            locks=Path("./locks"),
            distributed_requests=Path("./distributed_requests"),
        ),
        # Legacy postgres settings (kept for backward compatibility)
        # These are only used if you still have old sync() calls
        # postgres_host="localhost",
        # postgres_port=5432,
        # postgres_password="postgres",
    )