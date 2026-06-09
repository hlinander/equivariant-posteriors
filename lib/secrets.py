"""Decrypt secrets.enc.yaml via the `sops` CLI and cache the result.

Why a CLI shell-out instead of a Python library: pure-Python SOPS readers
exist but are thin / unmaintained, and SOPS's leaf-level format (per-value
AES-GCM with an age-wrapped data key) is what we want for reviewable diffs.
The cost is one fork + one parse per process, done lazily on first access.

Expected layout at repo root:
    .sops.yaml          # creation rules + age recipients
    secrets.enc.yaml    # encrypted, committed
    secrets.example.yaml  # plaintext skeleton

Bootstrap and runtime requirements:
    sops                in PATH (~/bin/sops works)
    age-keygen          to mint a keypair (once, per machine/user)
    age private key at  ~/.config/sops/age/keys.txt  (or SOPS_AGE_KEY_FILE)
"""
import functools
import json
import shutil
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENC_FILE = _REPO_ROOT / "secrets.enc.yaml"


@functools.cache
def load() -> dict:
    """Return the decrypted secrets dict. Cached per process."""
    if shutil.which("sops") is None:
        raise RuntimeError(
            "sops binary not found in PATH — install from "
            "https://github.com/getsops/sops/releases or `brew install sops`"
        )
    if not _ENC_FILE.is_file():
        raise FileNotFoundError(
            f"{_ENC_FILE} missing — see secrets.example.yaml + .sops.yaml for "
            "bootstrap steps"
        )
    try:
        out = subprocess.run(
            ["sops", "-d", "--output-type", "json", str(_ENC_FILE)],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"sops failed to decrypt {_ENC_FILE.name}: {e.stderr.strip()}\n"
            "Common cause: age private key is not at "
            "~/.config/sops/age/keys.txt and SOPS_AGE_KEY_FILE is not set."
        ) from e
    return json.loads(out)
