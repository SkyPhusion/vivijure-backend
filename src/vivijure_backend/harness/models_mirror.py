"""Cold-start model mirror: populate the worker's local HF cache from R2 at startup.

Why pull at cold start instead of baking weights into the image: the full set is ~hundreds of
GB; a layer that size is rejected by GHCR and ingests slowly, while R2 has no layer limit and
is fast and already seeded. So the image stays tiny and a cold worker mirrors what it needs
from `r2:<bucket>/models` with `rclone --links` (faithfully reconstructing the HF cache); a
warm worker sees the completion sentinel and skips. The same R2 token does double duty here
and for job I/O; the worker holds no other credential.

The sentinel is written only after the pull fully succeeds, so a worker killed mid-pull leaves
no marker and re-runs (rclone copy is idempotent) rather than rendering against a half-mirror.
The rclone command is built by a pure helper so it tests without spawning anything.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

# Repos kept in R2 as the canonical mirror but NOT pulled on cold start (lazy / not loaded by
# the default i2v path). Tune per the live model set; this is the deploy's call, not the code's.
DEFAULT_SKIP_REPOS = (
    "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers",
    "spaces--InstantX--InstantID",
)

# HF's abandoned download temp files; model-presence checks treat any *.incomplete as a broken
# repo, so never mirror them.
_INCOMPLETE_GLOB = "**/*.incomplete"

# Written under the models root only after the pull fully succeeds (see module docstring).
SENTINEL = ".vj-mirror-complete"


def rclone_conf(env: dict, conf_dir: Path) -> Path:
    """Write an rclone.conf for the R2 store from the worker's R2_* env. Raises if creds are
    incomplete so the worker fails here, loudly, not later at the model-presence gate."""
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT") if not env.get(k)]
    if missing:
        raise RuntimeError("models_mirror: incomplete R2 creds; missing: " + ", ".join(missing))
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf = conf_dir / "rclone.conf"
    conf.write_text(
        "[r2]\ntype = s3\nprovider = Cloudflare\n"
        f"access_key_id = {env['R2_ACCESS_KEY_ID']}\n"
        f"secret_access_key = {env['R2_SECRET_ACCESS_KEY']}\n"
        f"endpoint = {env['R2_ENDPOINT']}\n"
        "acl = private\nno_check_bucket = true\n"
    )
    conf.chmod(0o600)
    return conf


def mirror_cmd(conf: Path, src: str, dst: Path, *, skip_repos: tuple[str, ...] = ()) -> list[str]:
    """argv for one `rclone copy --links` mirror leg. Pure: built and asserted without rclone."""
    cmd = ["rclone", "--config", str(conf), "copy", "--links",
           "--transfers", "16", "--checkers", "16",
           "--stats", "60s", "--stats-one-line", "-v",
           "--exclude", _INCOMPLETE_GLOB]
    for repo in skip_repos:
        cmd += ["--exclude", f"hub/{repo}/**"]
    cmd += [src, str(dst)]
    return cmd


def ensure_models(*, env: dict | None = None, log: Callable[[str], None] = print,
                  skip_repos: tuple[str, ...] = DEFAULT_SKIP_REPOS) -> bool:
    """Mirror the kept model set from R2 into the local HF cache + antelopev2 dir.

    Returns True if a pull ran, False if it was skipped (warm worker, or no R2 creds so weights
    are assumed pre-provisioned). Raises on a hard failure (missing rclone, failed pull).
    """
    e = env if env is not None else os.environ
    hf_home = Path(e.get("HF_HOME", "/opt/models/hf-cache"))
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    bucket = e.get("R2_BUCKET", "vivijure")
    sentinel = models_root / SENTINEL

    if sentinel.exists():
        log("models_mirror: warm worker (sentinel present); skipping R2 pull.")
        return False
    if not e.get("R2_ACCESS_KEY_ID"):
        log("models_mirror: no R2 creds; assuming weights are pre-provisioned.")
        return False
    if shutil.which("rclone") is None:
        raise RuntimeError("models_mirror: rclone is not installed in the image")

    conf = rclone_conf(e, Path(tempfile.gettempdir()) / "vj-rclone")
    hf_home.mkdir(parents=True, exist_ok=True)
    log(f"models_mirror: cold worker -> mirroring r2:{bucket}/models to {hf_home} "
        f"(skipping {len(skip_repos)} lazy repos)...")
    subprocess.run(mirror_cmd(conf, f"r2:{bucket}/models/hf-cache", hf_home, skip_repos=skip_repos), check=True)

    antelope = models_root / "antelopev2"
    antelope.mkdir(parents=True, exist_ok=True)
    subprocess.run(mirror_cmd(conf, f"r2:{bucket}/models/antelopev2", antelope), check=True)

    models_root.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("ok\n")
    log("models_mirror: model mirror from R2 complete.")
    return True
