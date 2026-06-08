"""CPU tests for the GpuPipeline convergence: the pure config->params mappers, and the
execute() orchestration with the three GPU stages stubbed (no torch, no R2). Mirrors the
fake-stage pattern in tests/test_harness.py."""
import io
import json
import shutil
import tarfile
from pathlib import Path

import yaml

from vivijure_backend.config import RenderConfig
from vivijure_backend.contract import Bundle, RenderRequest, Scene
from vivijure_backend.harness.handler import Outputs, run_job
from vivijure_backend.orchestrator import plan as make_plan
from vivijure_backend.pipeline import GpuPipeline, i2v_params_from, keyframe_params_from
from vivijure_backend.routing import QualityTier


# ----------------------------------------------------------------- pure config -> params

def test_keyframe_params_final_is_full_step():
    p = keyframe_params_from(RenderConfig.for_tier(QualityTier.FINAL))
    assert p.few_step is False and p.steps == 30


def test_keyframe_params_draft_is_few_step():
    p = keyframe_params_from(RenderConfig.for_tier(QualityTier.DRAFT))
    assert p.few_step is True and p.steps == 4   # distill_steps, not the full-path steps


def test_keyframe_params_pull_multichar_scales():
    cfg = RenderConfig.from_request("final", {"keyframe": {"multi_char": {
        "lora_scale_per_slot": 0.25, "ip_adapter_scale_per_slot": 0.6,
        "max_slots": 2, "pose_conditioning": False}}})
    p = keyframe_params_from(cfg)
    assert p.lora_scale == 0.25
    assert p.ip_adapter_scale == 0.6
    assert p.pose_conditioning is False
    assert p.max_slots == 2


def test_i2v_params_track_tier_and_scene_duration():
    final = i2v_params_from(RenderConfig.for_tier(QualityTier.FINAL), Scene(prompt="x", target_seconds=4))
    assert final.distill is False and final.steps == 40
    assert final.num_frames == 65   # round(4*16)=64 -> snapped up to 4k+1
    draft = i2v_params_from(RenderConfig.for_tier(QualityTier.DRAFT), Scene(prompt="x", target_seconds=5))
    assert draft.distill is True and draft.steps == 4
    assert draft.num_frames == 81   # 5*16=80 -> snapped to 81 (and at the ceiling)


# ----------------------------------------------------------------- bundle + stub pipeline

STORYBOARD = {
    "title": "neon", "use_characters": ["A", "B"], "style_prefix": "anime,",
    "scenes": [
        {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
        {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
        {"id": "shot_03", "prompt": "A, authored", "character_slots": ["A"],
         "target_seconds": 3, "start_image": "injected/shot_03.png"},
    ],
}


def _extract_bundle(tmp_path: Path) -> Bundle:
    tarp = tmp_path / "b.tar.gz"
    members = {
        "storyboard.yaml": yaml.safe_dump(STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {
            "A": {"name": "Vesper", "prompt": "teal"}, "B": {"name": "Rhode", "prompt": "orange"}}}).encode(),
        "injected/shot_03.png": b"PNG-ish",   # the authored start_image for the INJECT shot
    }
    with tarfile.open(tarp, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return Bundle.extract(tarp, tmp_path / "project")


class StubPipeline(GpuPipeline):
    """GpuPipeline with the three GPU stages replaced by recording stubs that write empty
    artifact files. Exercises the orchestration without torch."""
    def __init__(self, config, pretrained_loras=None):
        super().__init__(config=config, pretrained_loras=pretrained_loras or {}, server=object())
        self.trained: list[str] = []
        self.keyframed: list[str] = []
        self.animated: list[str] = []
        self.keyframe_loras: dict[str, list[str]] = {}

    def _train_slot(self, char, out_dir):
        self.trained.append(char.slot)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / "lora.safetensors"; f.write_bytes(b"x"); return f

    def _render_keyframe(self, scene, cast, storyboard, out_path, lora_paths):
        self.keyframed.append(scene.id)
        self.keyframe_loras[scene.id] = sorted(lora_paths)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x"); return out_path

    def _animate(self, scene, keyframe_path, prompt, out_path):
        assert Path(keyframe_path).exists(), "animating from a keyframe that was never staged"
        self.animated.append(scene.id)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x"); return out_path


# --------------------------------------------------------------------- execute orchestration

def test_execute_trains_keyframes_and_animates_the_whole_plan(tmp_path):
    bundle = _extract_bundle(tmp_path)
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "final"})
    plan = make_plan(req, bundle.storyboard)
    pipe = StubPipeline(req.config)
    out = pipe.execute(plan, bundle, tmp_path / "work")

    assert sorted(pipe.trained) == ["A", "B"]                 # nothing pretrained -> train both
    assert sorted(out.loras) == ["A", "B"]
    assert sorted(out.keyframes) == ["shot_01", "shot_02"]    # shot_03 is INJECT, not generated
    assert sorted(pipe.keyframed) == ["shot_01", "shot_02"]
    assert sorted(pipe.animated) == ["shot_01", "shot_02", "shot_03"]  # all needs_i2v, inject staged
    assert [s for s, _ in out.clips] == ["shot_01", "shot_02", "shot_03"]
    # keyframing saw the freshly trained adapters
    assert pipe.keyframe_loras["shot_02"] == ["A", "B"]


def test_execute_honors_reuse_and_pretrained(tmp_path):
    bundle = _extract_bundle(tmp_path)
    # A is pretrained (skip training, feed a staged adapter); shot_01 keyframe already exists.
    pre = tmp_path / "preA.safetensors"; pre.write_bytes(b"x")
    req = RenderRequest.from_dict({"action": "render", "project": "neon", "bundle_key": "x",
                                   "quality_tier": "final", "pretrained_loras": {"A": str(pre)}})
    plan = make_plan(req, bundle.storyboard,
                     trained_slots=set(req.pretrained_loras), existing_keyframes={"shot_01"})
    work = tmp_path / "work"
    (work / "keyframes").mkdir(parents=True)
    (work / "keyframes" / "shot_01.png").write_bytes(b"x")   # stage the reused keyframe

    pipe = StubPipeline(req.config, pretrained_loras=req.pretrained_loras)
    out = pipe.execute(plan, bundle, work)

    assert pipe.trained == ["B"]                               # A pretrained -> only B trains
    assert "shot_01" not in pipe.keyframed                     # shot_01 reused, not regenerated
    assert sorted(out.keyframes) == ["shot_02"]
    assert "shot_01" in pipe.animated                          # reused keyframe still animates
    # the pretrained A adapter was wired into keyframing
    assert "A" in pipe.keyframe_loras["shot_02"]


def test_execute_skips_i2v_when_reused_keyframe_is_missing(tmp_path):
    # A REUSE shot whose keyframe was never staged is skipped, not crashed.
    bundle = _extract_bundle(tmp_path)
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "final"})
    plan = make_plan(req, bundle.storyboard, existing_keyframes={"shot_01"})
    pipe = StubPipeline(req.config)
    out = pipe.execute(plan, bundle, tmp_path / "work")   # shot_01 keyframe not staged
    assert "shot_01" not in pipe.animated
    assert "shot_01" not in [s for s, _ in out.clips]


# --------------------------------------------------------------------- run_job end to end

class FakeStore:
    def __init__(self, bundle_tar: Path):
        self.bundle_tar = bundle_tar; self.puts: list[str] = []; self.tars: list[str] = []

    def get_file(self, key, dest):
        shutil.copy(self.bundle_tar, dest); return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists(); self.puts.append(key); return key

    def put_dir_as_tar(self, src_dir, key, *, metadata=None):
        self.tars.append(key); return key


def test_run_job_drives_gpu_pipeline_offloaded(tmp_path):
    # The whole harness flow on CPU with a stubbed GpuPipeline: plan -> execute -> finish.
    _extract_bundle(tmp_path)  # writes the bundle tar at tmp_path/b.tar.gz
    store = FakeStore(tmp_path / "b.tar.gz")
    pipe = StubPipeline(RenderConfig.for_tier(QualityTier.FINAL))

    res = run_job(
        {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
         "quality_tier": "final", "render_overrides": {"finish_offloaded": True}},
        pipeline=pipe, store=store, workdir=tmp_path / "work")

    assert res["lora"]["A"]["lora_id"].endswith("A/pytorch_lora_weights.safetensors")
    assert [c["shot_id"] for c in res["clips"]] == ["shot_01", "shot_02", "shot_03"]
    assert {k["shot_id"] for k in res["keyframes"]} == {"shot_01", "shot_02"}
    assert any(k.endswith("manifest.json") for k in store.puts)
    assert res["state_key"] == "projects/neon/state.tar.gz"


# ----------------------------------------------------- pretrained-LoRA R2 staging (item B)

class StagingStore(FakeStore):
    """FakeStore that records the keys it serves, so a test can assert a LoRA was fetched."""
    def __init__(self, bundle_tar):
        super().__init__(bundle_tar); self.gets: list[str] = []

    def get_file(self, key, dest):
        self.gets.append(key)
        return super().get_file(key, dest)


def test_run_job_stages_pretrained_lora_from_r2_and_skips_training(tmp_path):
    # A render that reuses a slot's R2 LoRA must NOT retrain it, and the adapter must be pulled
    # to local disk (the GPU layer never touches R2) and fed to keyframing.
    _extract_bundle(tmp_path)
    store = StagingStore(tmp_path / "b.tar.gz")
    LORA_KEY = "loras/neon/A/pytorch_lora_weights.safetensors"
    pipe = StubPipeline(RenderConfig.for_tier(QualityTier.DRAFT), pretrained_loras={"A": LORA_KEY})

    res = run_job(
        {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
         "quality_tier": "draft", "pretrained_loras": {"A": LORA_KEY},
         "render_overrides": {"finish_offloaded": True}},  # skip the ffmpeg merge of stub clips
        pipeline=pipe, store=store, workdir=tmp_path / "work", job_id="j")

    assert pipe.trained == ["B"]                              # A reused -> only B trains
    assert LORA_KEY in store.gets                             # the adapter was actually fetched
    # the pipeline now holds a LOCAL staged path for A, not the R2 key, and it exists on disk
    assert "/pretrained/A/" in pipe.pretrained_loras["A"]
    assert Path(pipe.pretrained_loras["A"]).is_file()
    assert "A" in pipe.keyframe_loras["shot_01"]              # the staged LoRA reached keyframing
    assert res["lora"]["A"]["lora_id"] == LORA_KEY            # result still reports the durable R2 key


def test_run_job_fails_fast_when_a_reused_lora_cannot_be_staged(tmp_path):
    # A requested-but-unfetchable LoRA must fail the job BEFORE any GPU work, not silently render
    # the character without its identity.
    import pytest
    from vivijure_backend.harness.handler import HarnessError

    _extract_bundle(tmp_path)

    class MissingLoraStore(FakeStore):
        def get_file(self, key, dest):
            if str(key).endswith(".tar.gz"):
                return super().get_file(key, dest)
            raise FileNotFoundError(key)                      # the LoRA key is not in R2

    pipe = StubPipeline(RenderConfig.for_tier(QualityTier.DRAFT),
                        pretrained_loras={"A": "loras/x/A.safetensors"})
    with pytest.raises(HarnessError, match="could not stage pretrained LoRA"):
        run_job({"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
                 "quality_tier": "draft", "pretrained_loras": {"A": "loras/x/A.safetensors"}},
                pipeline=pipe, store=MissingLoraStore(tmp_path / "b.tar.gz"),
                workdir=tmp_path / "work", job_id="j")
    assert pipe.trained == []                                 # failed before training anything
