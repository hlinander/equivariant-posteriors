from pathlib import Path

from lib.compute_env_config import ComputeEnvironment, Paths
from lib.analytics_config import (
    AnalyticsConfig,
    StagingFilesystem,
    CentralDuckDB,
)

# Base directory for all local data
LOCAL_DIR = Path(".local")


def get_analytics_config():
    """Configure analytics pipeline (staging and central database)."""
    return AnalyticsConfig(
        export_interval_seconds=5,
        ingest_interval_seconds=5,
        staging=StagingFilesystem(
            staging_dir=LOCAL_DIR / "staging",
            archive_dir=LOCAL_DIR / "archive",
        ),
        central=CentralDuckDB(
            db_path=LOCAL_DIR / "analytics.db",
        ),
    )


def get_env() -> ComputeEnvironment:
    """
    Configure training client settings.
    """

    print("Getting compute environment")

    return ComputeEnvironment(
        paths=Paths(
            checkpoints=Path("/proj/heal_pangu/eqp/checkpoints"),
            artifacts=Path("/proj/heal_pangu/eqp/artifacts"),
            datasets=Path("/proj/heal_pangu/datasets"),
            locks=Path("./locks"),
            distributed_requests=Path("./distributed_requests"),
        ),
    )
