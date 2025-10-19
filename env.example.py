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
    # Option 1: Local Development (Filesystem + Local DB)
    # ---------------------------------------------------------------------------
    # Perfect for: Local development, single-machine testing
    # return AnalyticsConfig(
    #     staging=StagingFilesystem(
    #         staging_dir=Path("./staging"),
    #         archive_dir=Path("./archive"),
    #     ),
    #     central=CentralDuckDB(
    #         db_path=Path("./central.db")
    #     ),
    #     export_interval_seconds=60,  # Export every minute (faster for dev)
    #     ingest_interval_seconds=60,
    # )

    # ---------------------------------------------------------------------------
    # Option 2: Local Development (MinIO + Local DB)
    # ---------------------------------------------------------------------------
    # Perfect for: Testing S3 integration locally with MinIO
    s3_minio = S3Config(
        key="minioadmin",
        secret="minioadmin",
        region="us-east-1",
        endpoint="http://localhost:9000",
    )
    return AnalyticsConfig(
        staging=StagingS3(
            s3=s3_minio,
            bucket="metrics-staging",
            prefix="staging",
            archive_prefix="archive",
        ),
        central=CentralDuckDB(
            db_path=Path("./central.db")
        ),
        export_interval_seconds=60,
        ingest_interval_seconds=60,
    )

    # ---------------------------------------------------------------------------
    # Option 3: Production HPC (S3 + DuckLake)
    # ---------------------------------------------------------------------------
    # Perfect for: HPC cluster with shared central database
    # Note: Same S3 config used for both staging and DuckLake storage
    # s3_prod = S3Config(
    #     key="YOUR_S3_ACCESS_KEY",
    #     secret="YOUR_S3_SECRET_KEY",
    #     region="us-east-1",
    #     endpoint="https://s3.amazonaws.com",
    # )
    # postgres_prod = PostgresConfig(
    #     host="postgres.hpc.internal",
    #     port=5432,
    #     user="postgres",
    #     password="YOUR_POSTGRES_PASSWORD",
    #     dbname="ducklake",
    # )
    # return AnalyticsConfig(
    #     staging=StagingS3(
    #         s3=s3_prod,
    #         bucket="hpc-metrics-staging",
    #         prefix="staging",
    #         archive_prefix="archive",
    #     ),
    #     central=CentralDuckLake(
    #         postgres=postgres_prod,
    #         s3=s3_prod,  # Same S3 config!
    #         data_path="s3://hpc-ducklake-data",
    #     ),
    #     export_interval_seconds=300,  # Export every 5 minutes
    #     ingest_interval_seconds=300,
    # )

    # ---------------------------------------------------------------------------
    # Option 4: Shared Filesystem (e.g., NFS/Lustre + DuckLake)
    # ---------------------------------------------------------------------------
    # Perfect for: HPC with shared filesystem, no S3 for staging
    # Note: DuckLake still needs S3 for its data path
    # s3_ducklake = S3Config(
    #     key="YOUR_S3_ACCESS_KEY",
    #     secret="YOUR_S3_SECRET_KEY",
    #     endpoint="https://s3.amazonaws.com",
    # )
    # postgres_hpc = PostgresConfig(
    #     host="postgres.hpc.internal",
    #     port=5432,
    #     password="YOUR_POSTGRES_PASSWORD",
    # )
    # return AnalyticsConfig(
    #     staging=StagingFilesystem(
    #         staging_dir=Path("/shared/metrics/staging"),
    #         archive_dir=Path("/shared/metrics/archive"),
    #     ),
    #     central=CentralDuckLake(
    #         postgres=postgres_hpc,
    #         s3=s3_ducklake,
    #         data_path="s3://hpc-ducklake-data",
    #     ),
    #     export_interval_seconds=300,
    #     ingest_interval_seconds=300,
    # )


# ============================================================================
# Training Client Configuration (Paths, etc.)
# ============================================================================

def get_env() -> ComputeEnvironment:
    """
    Configure training client settings.

    This is separate from analytics config - it's for training-specific
    settings like checkpoint paths, dataset paths, etc.
    """
    return ComputeEnvironment(
        paths=Paths(
            checkpoints=Path("./checkpoints"),
            datasets=Path("./datasets"),
            locks=Path("./locks"),
            distributed_requests=Path("./distributed_requests"),
            artifacts=Path("./artifacts"),
        ),
        # Legacy postgres settings (kept for backward compatibility)
        # These are only used if you still have old sync() calls
        postgres_host="localhost",
        postgres_port=5432,
        postgres_password="postgres",
    )


# ============================================================================
# Common Patterns
# ============================================================================

# Pattern 1: Environment Variable Overrides
# -----------------------------------------
# import os
#
# def get_analytics_config() -> AnalyticsConfig:
#     if os.getenv("USE_S3") == "true":
#         return AnalyticsConfig(
#             staging=StagingS3(
#                 bucket=os.getenv("S3_BUCKET", "metrics"),
#                 key=os.getenv("S3_KEY"),
#                 secret=os.getenv("S3_SECRET"),
#                 endpoint=os.getenv("S3_ENDPOINT"),
#             ),
#             central=CentralDuckDB()
#         )
#     else:
#         return AnalyticsConfig(
#             staging=StagingFilesystem(),
#             central=CentralDuckDB()
#         )

# Pattern 2: Multi-Environment Support
# -------------------------------------
# import os
#
# def get_analytics_config() -> AnalyticsConfig:
#     env = os.getenv("DEPLOYMENT_ENV", "local")
#
#     if env == "local":
#         return AnalyticsConfig(
#             staging=StagingFilesystem(),
#             central=CentralDuckDB()
#         )
#     elif env == "staging":
#         return AnalyticsConfig(
#             staging=StagingS3(...),
#             central=CentralDuckDB()
#         )
#     elif env == "production":
#         return AnalyticsConfig(
#             staging=StagingS3(...),
#             central=CentralDuckLake(...)
#         )

# Pattern 3: Conditional DuckLake
# --------------------------------
# Only use DuckLake if postgres is available, otherwise local DB
#
# import os
#
# def get_analytics_config() -> AnalyticsConfig:
#     postgres_host = os.getenv("POSTGRES_HOST")
#
#     if postgres_host:
#         central = CentralDuckLake(
#             postgres_host=postgres_host,
#             postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
#             postgres_password=os.getenv("POSTGRES_PASSWORD"),
#         )
#     else:
#         central = CentralDuckDB()
#
#     return AnalyticsConfig(
#         staging=StagingS3(...),
#         central=central
#     )
