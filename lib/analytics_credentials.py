"""Load analytics S3 staging credentials from a file outside the repo.

Why out-of-tree: credentials must never be committed. `.gitignore` is only
advisory — it does not stop `git add -f`, and it does nothing about a key
pasted into a tracked `.py`. A file that lives in the user's home config dir
is simply not in the working tree, so a force-add cannot reach it. That makes
the accidental-checkin failure mode structurally impossible, with no
encryption ceremony for the many users who each hold their own credentials.

Default location:
    ~/.config/equivariant-posteriors/analytics.yaml   (override: ANALYTICS_CREDENTIALS_FILE)

File shape (see analytics.example.yaml):
    s3:
      key:    ...            # required
      secret: ...            # required
      endpoint: https://...  # required
      bucket: metrics-staging  # required
      region: us-east-1      # optional, default us-east-1
      url_style: path        # optional, default path  (path | vhost)
      prefix: staging        # optional, default staging
      archive_prefix: archive  # optional, default archive
"""
import functools
import os
from pathlib import Path

import yaml

from lib.analytics_config import S3Config, StagingS3


DEFAULT_PATH = Path.home() / ".config" / "equivariant-posteriors" / "analytics.yaml"
_REQUIRED = ("key", "secret", "endpoint", "bucket")


def credentials_path() -> Path:
    """Path to the credentials file (ANALYTICS_CREDENTIALS_FILE overrides)."""
    override = os.environ.get("ANALYTICS_CREDENTIALS_FILE")
    return Path(override).expanduser() if override else DEFAULT_PATH


@functools.cache
def _read() -> dict:
    """Read + validate the `s3:` block. Cached per process."""
    path = credentials_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Analytics credentials not found at {path}.\n"
            "Copy analytics.example.yaml there and fill in your S3 key/secret "
            "(see the 'Analytics credentials' section of the README)."
        )
    s3 = (yaml.safe_load(path.read_text()) or {}).get("s3") or {}
    missing = [k for k in _REQUIRED if not s3.get(k)]
    if missing:
        raise ValueError(
            f"{path}: missing required s3 field(s): {', '.join(missing)}"
        )
    return s3


def s3_config() -> S3Config:
    """Just the credential/connection triple — reusable for DuckLake central."""
    s3 = _read()
    return S3Config(
        key=s3["key"],
        secret=s3["secret"],
        endpoint=s3["endpoint"],
        region=s3.get("region", "us-east-1"),
        url_style=s3.get("url_style", "path"),
    )


def load_staging_s3() -> StagingS3:
    """Full S3 staging config (credentials + bucket/prefixes) for env.py."""
    s3 = _read()
    return StagingS3(
        s3=s3_config(),
        bucket=s3["bucket"],
        prefix=s3.get("prefix", "staging"),
        archive_prefix=s3.get("archive_prefix", "archive"),
    )
