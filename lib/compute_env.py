import traceback
from pathlib import Path

from lib.compute_env_config import ComputeEnvironment


_current_env = None

_TERM_ENDC = "\033[0m"
_TERM_GREEN = "\033[92m"
_TERM_WARNING = "\033[93m"

_DEFAULT_ENV_TEMPLATE = '''"""
Local compute environment configuration.
Modify paths and settings as needed for your setup.

All data is stored under .local/ by default - add to .gitignore:
    .local/
"""
from pathlib import Path

from lib.compute_env_config import ComputeEnvironment, Paths
from lib.analytics_config import (
    AnalyticsConfig,
    StagingFilesystem,
    CentralDuckDB,
)
from lib.slurm import SlurmConfig

# Base directory for all local data
LOCAL_DIR = Path(".local")


def get_env():
    """Configure local paths for checkpoints, artifacts, etc."""
    return ComputeEnvironment(
        paths=Paths(
            checkpoints=LOCAL_DIR / "checkpoints",
            locks=LOCAL_DIR / "locks",
            distributed_requests=LOCAL_DIR / "distributed_requests",
            artifacts=LOCAL_DIR / "artifacts",
            datasets=LOCAL_DIR / "datasets",
        ),
    )


def get_slurm_config():
    """Configure SLURM job parameters for sweep submission."""
    return SlurmConfig(
        time="24:00:00",
        gpus=1,
    )


def get_analytics_config():
    """Configure analytics pipeline (staging and central database)."""
    return AnalyticsConfig(
        staging=StagingFilesystem(
            staging_dir=LOCAL_DIR / "staging",
            archive_dir=LOCAL_DIR / "archive",
        ),
        central=CentralDuckDB(
            db_path=LOCAL_DIR / "analytics.db",
        ),
    )
'''


def _create_default_env():
    env_path = Path("env.py")
    if not env_path.exists():
        print(f"{_TERM_GREEN}[Compute environment] Creating default env.py{_TERM_ENDC}")
        env_path.write_text(_DEFAULT_ENV_TEMPLATE)


def env():
    global _current_env
    if _current_env is None:
        _create_default_env()
        try:
            import env

            _current_env = env.get_env()
        except Exception:
            print(
                f"{_TERM_WARNING}[Compute environment] Could not load env.py: \n{traceback.format_exc()}{_TERM_ENDC}"
            )
            print(f"{_TERM_WARNING}[Compute environment] Using defaults{_TERM_ENDC}")
            _current_env = ComputeEnvironment()

        for key, value in _current_env.__dict__.items():
            print(f"{_TERM_GREEN}[Compute environment] {key}: {str(value)}{_TERM_ENDC}")

    return _current_env


env()
