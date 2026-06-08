"""The harness is CPU-only by design: keys/mirror/config are pure, and the whole job flow runs
against a fake pipeline + fake object store, so `run_job` is tested without a GPU, R2, or (on
the offloaded path) ffmpeg."""
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from vivijure_backend.harness import keys
from vivijure_backend.harness.handler import HarnessError, Outputs, run_job
from vivijure_backend.harness.models_mirror import mirror_cmd, rclone_conf
from vivijure_backend.harness.r2 import R2Config


# ------------------------------------------------------------------------------- keys

def test_keys_layout():
    assert keys.output_key("neon") == "renders/neon/full.mp4"
    assert keys.state_key("neon") == "projects/neon/state.tar.gz"
    assert keys.lora_key("neon", "A") == "loras/neon/A/pytorch_lora_weights.safetensors"
    assert keys.keyframe_key("neon", "shot_01") == "renders/neon/keyframes/shot_01.png"
    assert keys.clip_key("neon", "shot_02") == "renders/neon/clips/shot_02.mp4"


def test_key_slug_is_path_safe():
    # a name with spaces/slashes must not smuggle extra path segments into a key
    assert keys.output_key("neon rain/standoff") == "renders/neon_rain_standoff/full.mp4"
    assert keys.output_key("   ") == "renders/untitled/full.mp4"


# --------------------------------------------------------------------- models mirror

def test_mirror_cmd_copies_with_links_and_excludes():
    cmd = mirror_cmd(Path("/c.conf"), "r2:vivijure/models/hf-cache", Path("/hf"),
                     skip_repos=("models--X", "spaces--Y"))
    assert cmd[:5] == ["rclone", "--config", "/c.conf", "copy", "--links"]
    assert "--exclude" in cmd and "**/*.incomplete" in cmd
    assert "hub/models--X/**" in cmd and "hub/spaces--Y/**" in cmd
    assert cmd[-2:] == ["r2:vivijure/models/hf-cache", "/hf"]  # src, dst last


def test_rclone_conf_writes_creds_and_rejects_partial(tmp_path):
    conf = rclone_conf({"R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s",
                        "R2_ENDPOINT": "https://x.r2"}, tmp_path)
    text = conf.read_text()
    assert "access_key_id = k" in text and "endpoint = https://x.r2" in text
    with pytest.raises(RuntimeError, match="incomplete R2 creds"):
        rclone_conf({"R2_ACCESS_KEY_ID": "k"}, tmp_path)


def test_r2config_from_env_validates():
    cfg = R2Config.from_env({"R2_ENDPOINT": "e", "R2_ACCESS_KEY_ID": "k",
                             "R2_SECRET_ACCESS_KEY": "s", "R2_BUCKET": "vivijure"})
    assert cfg.bucket == "vivijure"
    with pytest.raises(RuntimeError, match="missing env"):
        R2Config.from_env({"R2_ENDPOINT": "e"})


# ----------------------------------------------------------- fakes for the job flow

STORYBOARD = {
    "title": "neon", "use_characters": ["A", "B"],
    "scenes": [
        {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
        {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
    ],
}


def _bundle_tar(path: Path) -> Path:
    members = {
        "storyboard.yaml": yaml.safe_dump(STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {
            "A": {"name": "Vesper", "prompt": "teal"}, "B": {"name": "Rhode", "prompt": "orange"}}}).encode(),
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


class FakeStore:
    """Records puts; serves the prebuilt bundle for any get. No network."""
    def __init__(self, bundle_tar: Path):
        self.bundle_tar = bundle_tar
        self.puts: list[str] = []
        self.tars: list[str] = []

    def get_file(self, key, dest):
        shutil.copy(self.bundle_tar, dest)
        return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists(), f"uploading a nonexistent file: {path}"
        self.puts.append(key)
        return key

    def put_dir_as_tar(self, src_dir, key, *, metadata=None):
        self.tars.append(key)
        return key


class FakePipeline:
    """Produces empty artifact files for exactly the work the plan kept; no GPU."""
    def execute(self, plan, bundle, workdir):
        out = Outputs()
        for slot in plan.lora.train:
            p = workdir / f"lora_{slot}.safetensors"; p.write_bytes(b"x"); out.loras[slot] = p
        for s in plan.scenes:
            if s.keyframe_mode.value == "generate":
                p = workdir / f"{s.shot_id}.png"; p.write_bytes(b"x"); out.keyframes[s.shot_id] = p
            if s.needs_i2v:
                p = workdir / f"{s.shot_id}.mp4"; p.write_bytes(b"x"); out.clips.append((s.shot_id, p))
        return out


def _job(**over):
    return {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
            "quality_tier": "final", **over}


# --------------------------------------------------------------------- job flow

def test_run_job_offloaded_emits_clips_and_manifest(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(
        _job(render_overrides={"finish_offloaded": True}, pretrained_loras={"A": "loras/ext/A.safetensors"}),
        pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")

    # pretrained A reused, B trained+uploaded
    assert res["lora"]["A"] == {"lora_id": "loras/ext/A.safetensors"}
    assert res["lora"]["B"]["lora_id"] == "loras/neon/B/pytorch_lora_weights.safetensors"
    # offloaded: per-shot clips in storyboard order, a manifest, and NO merged output
    assert [c["shot_id"] for c in res["clips"]] == ["shot_01", "shot_02"]
    assert res["output_key"] is None
    assert any(k.endswith("manifest.json") for k in store.puts)
    assert res["state_key"] == "projects/neon/state.tar.gz"
    # keyframes uploaded for both generated shots
    assert {k["shot_id"] for k in res["keyframes"]} == {"shot_01", "shot_02"}


def test_run_job_rejects_invalid_storyboard(tmp_path):
    bad = dict(STORYBOARD, use_characters=["A"])  # shot_02 references B, not in use_characters
    tarp = tmp_path / "bad.tar.gz"
    with tarfile.open(tarp, "w:gz") as tf:
        for name, data in {"storyboard.yaml": yaml.safe_dump(bad).encode(),
                           "characters/registry.json": b'{"characters":{}}'}.items():
            info = tarfile.TarInfo(name=name); info.size = len(data); tf.addfile(info, io.BytesIO(data))
    with pytest.raises(HarnessError, match="invalid render job"):
        run_job(_job(), pipeline=FakePipeline(), store=FakeStore(tarp), workdir=tmp_path / "w")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_run_job_normal_merges_to_output_key(tmp_path):
    import subprocess

    class RealClipPipeline(FakePipeline):
        def execute(self, plan, bundle, workdir):
            out = super().execute(plan, bundle, workdir)
            real = []  # replace the empty stub clips with tiny real mp4s so assemble can merge
            for shot_id, _ in out.clips:
                p = workdir / f"{shot_id}_real.mp4"
                subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1:r=24",
                                "-pix_fmt", "yuv420p", "-t", "1", str(p)], capture_output=True, check=True)
                real.append((shot_id, p))
            out.clips = real
            return out

    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(), pipeline=RealClipPipeline(), store=store, workdir=tmp_path / "work")
    assert res["output_key"] == "renders/neon/full.mp4"
    assert res["seconds"] == pytest.approx(2.0, abs=0.4)  # two 1s clips merged
    assert "renders/neon/full.mp4" in store.puts
