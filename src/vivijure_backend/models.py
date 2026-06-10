"""Capability-aware model loading.

Turns `device.py`'s card facts into actual model loads in the right precision, and keeps
them resident: this is a persistent server, each model role loads once into VRAM and is
reused across renders, so only the first render on a fresh worker pays load time.

Clean-room: every loader targets the model's own public API (diffusers, torchao, the model
hubs) built from those docs, not from any prior pipeline.

Quant reality (verified June 2026):
  - SDXL is a UNet. The 4-bit engines (Nunchaku/SVDQuant) only cover DiTs (FLUX, Qwen-Image,
    SANA, Z-Image), so SDXL tops out at fp8 (MXFP8 on Blackwell, plain fp8 on Hopper) via
    torchao. NVFP4 is NOT available for SDXL.
  - A DiT keyframe model (FLUX/Qwen-Image) WOULD unlock NVFP4 (~1.7x on B200), at the cost of
    re-homing the identity stack (LoRA training, InstantID, IP-Adapter) off the SDXL ecosystem.
    That trade is `ModelFamily.DIT`; left as a deliberate future option.
  - Wan i2v is a video DiT: fp8 on both archs (4-bit-for-video is still young).

Heavy imports (torch / diffusers / torchao) are deferred into the load methods so this module
imports and unit-tests on a CPU box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .device import Device, Quant, current


class ModelRole(str, Enum):
    KEYFRAME_BASE = "keyframe_base"      # SDXL checkpoint that draws the still
    KEYFRAME_FEWSTEP = "keyframe_fewstep"  # distill LoRA/unet (Hyper-SD / DMD2) for 4-8 step
    I2V = "i2v"                          # Wan 2.2 image-to-video
    I2V_DISTILL = "i2v_distill"          # Wan2.2-Lightning distill LoRA (few-step i2v)
    INSTANTID = "instantid"              # face identity (insightface + ControlNet)
    IP_ADAPTER = "ip_adapter"            # per-slot identity conditioning
    CONTROLNET_POSE = "controlnet_pose"  # OpenPose: separates bodies in multi-char shots


class ModelFamily(str, Enum):
    SDXL_UNET = "sdxl_unet"  # fp8 ceiling (no 4-bit engine)
    DIT = "dit"              # FLUX/Qwen/SANA: NVFP4-capable on Blackwell
    VIDEO_DIT = "video_dit"  # Wan: fp8 on both archs
    AUX = "aux"              # adapters / controlnets / detectors: load at base dtype

    @property
    def fp4_capable(self) -> bool:
        return self is ModelFamily.DIT


@dataclass(frozen=True)
class ModelSpec:
    role: ModelRole
    repo_id: str
    family: ModelFamily
    subfolder: str | None = None
    note: str = ""
    weight_name: str | None = None   # specific file within repo_id (e.g. a per-step LoRA variant)


# Default model set. These are public Hugging Face repo ids; swap per project via overrides.
DEFAULT_SPECS: dict[ModelRole, ModelSpec] = {
    ModelRole.KEYFRAME_BASE: ModelSpec(
        ModelRole.KEYFRAME_BASE, "SG161222/RealVisXL_V5.0", ModelFamily.SDXL_UNET,
        note="photoreal-leaning SDXL default; anime alt cagliostrolab/animagine-xl-4.0",
    ),
    ModelRole.KEYFRAME_FEWSTEP: ModelSpec(
        ModelRole.KEYFRAME_FEWSTEP, "ByteDance/Hyper-SD", ModelFamily.AUX,
        weight_name="Hyper-SDXL-8steps-lora.safetensors",
        note="few-step distill LoRA (Hyper-SDXL); the 8-step variant is forgiving and still renders "
             "well at 4-6 steps, so one warm adapter serves both draft (4) and standard (8). DMD2 is "
             "the alt. Keyframes get animated, so 4-8 steps is plenty",
    ),
    ModelRole.I2V: ModelSpec(
        ModelRole.I2V, "Wan-AI/Wan2.2-I2V-A14B-Diffusers", ModelFamily.VIDEO_DIT,
    ),
    ModelRole.I2V_DISTILL: ModelSpec(
        ModelRole.I2V_DISTILL, "lightx2v/Wan2.2-Lightning", ModelFamily.AUX,
        note="4-step distill LoRA; watch diffusers LoRA-load compat (issue #12535), LightX2V/DiffSynth is the fallback loader",
    ),
    ModelRole.INSTANTID: ModelSpec(ModelRole.INSTANTID, "InstantX/InstantID", ModelFamily.AUX),
    ModelRole.IP_ADAPTER: ModelSpec(ModelRole.IP_ADAPTER, "h94/IP-Adapter", ModelFamily.AUX),
    ModelRole.CONTROLNET_POSE: ModelSpec(
        ModelRole.CONTROLNET_POSE, "xinsir/controlnet-openpose-sdxl-1.0", ModelFamily.AUX,
    ),
}


def quant_for(family: ModelFamily, device: Device) -> Quant:
    """The actual precision to load `family` at on `device`. The card sets the ceiling; the
    model family narrows it (SDXL has no 4-bit engine, so it never gets NVFP4 even on a card
    that supports it)."""
    if family is ModelFamily.AUX:
        return Quant.BF16  # adapters / LoRAs / detectors attach at the base dtype, not quantized
    if family.fp4_capable and device.supports_fp4:
        return Quant.NVFP4
    if device.supports_fp8:
        return Quant.FP8  # MXFP8 on Blackwell, plain fp8 on Hopper (torchao picks the variant)
    return Quant.BF16


class ModelServer:
    """Lazy, persistent model registry. Each role loads once and is cached for the process
    lifetime (the warm worker reuses it). `device` defaults to the live card."""

    def __init__(self, device: Device | None = None, specs: dict[ModelRole, ModelSpec] | None = None):
        self.device = device or current()
        self.specs = {**DEFAULT_SPECS, **(specs or {})}
        self._cache: dict[str, Any] = {}

    def plan(self) -> dict[str, str]:
        """The precision each role WILL load at on this card, no GPU touched. Drives logging
        and lets tests assert the matrix."""
        return {
            role.value: quant_for(spec.family, self.device).value
            for role, spec in self.specs.items()
        }

    # ---- loaders (deferred heavy imports; bodies need a GPU, verified on the pod) ----

    def keyframe_pipeline(self):
        """SDXL pipeline for keyframes: load at bf16, set the attention backend. Adapters and the
        per-scene character LoRAs (plus InstantID / IP-Adapter / ControlNet) are attached by
        keyframe.py per scene.

        NOT fp8-quantized: keyframe.py loads DYNAMIC per-scene character LoRAs onto the UNet, and
        peft cannot attach a LoRA to a torchao-quantized linear (TorchaoLoraLinear init mismatch,
        the #12535 family). SDXL is small (~6.5GB UNet), so bf16 is essentially free and keeps LoRA
        loading working on every card. (i2v CAN use fp8 because it FUSES its single distill LoRA
        before quantizing -- see i2v_pipeline.)"""
        if "keyframe" in self._cache:
            return self._cache["keyframe"]
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        spec = self.specs[ModelRole.KEYFRAME_BASE]
        cn_spec = self.specs[ModelRole.CONTROLNET_POSE]
        # The keyframe pipe is a ControlNet pipeline so the regional multi-character path can plant
        # two distinct bodies with an OpenPose skeleton (keyframe._render_regional); the single path
        # hands it a blank control image at conditioning_scale 0.0, which makes the ControlNet inert
        # (zero residual), so single-subject renders behave exactly like plain SDXL. Sharing one pipe
        # keeps the dynamic per-scene LoRA + IP-Adapter attach points identical across both paths.
        controlnet = ControlNetModel.from_pretrained(cn_spec.repo_id, torch_dtype=torch.bfloat16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            spec.repo_id, controlnet=controlnet, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        _set_attention(pipe, self.device)

        # Stash the pristine (full-step) scheduler so keyframe._apply_scheduler can restore it for the
        # final tier after a draft/standard render swapped in DDIM-trailing on this warm shared pipe.
        pipe._vj_default_scheduler = pipe.scheduler
        # Load the Hyper-SD few-step distill LoRA as a persistent BASE adapter (named "distill", NOT
        # fused -- unlike i2v, the keyframe pipe stays bf16 so dynamic per-scene character LoRAs can
        # still attach). keyframe._bind_loras keeps it active at weight 1.0 on draft/standard and 0.0
        # on final, so one warm pipe serves every tier.
        self._attach_keyframe_distill(pipe)

        self._cache["keyframe"] = pipe
        return pipe

    def face_analyzer(self):
        """insightface FaceAnalysis on the antelopev2 pack (mirrored to <VJ_MODELS_ROOT>/antelopev2
        on cold start) for InstantID: yields a face embedding + 5 keypoints from a reference face.
        insightface resolves a pack at `<root>/models/<name>`, so its root is the PARENT of the
        mirror's models_root (antelopev2 sits one level up from `<root>/models/`). Cached; deferred
        imports keep this module CPU-importable; validated on a pod (insightface path + onnxruntime
        provider selection are environment-finicky)."""
        if "face_analyzer" in self._cache:
            return self._cache["face_analyzer"]
        import os
        from insightface.app import FaceAnalysis

        models_root = os.environ.get("VJ_MODELS_ROOT", "/opt/models")
        root = os.path.dirname(models_root.rstrip("/")) or "/"
        app = FaceAnalysis(name="antelopev2", root=root,
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._cache["face_analyzer"] = app
        return app

    def instantid_pipeline(self):
        """InstantID single-character pipeline: the InstantID face-keypoints ControlNet on a fresh
        SDXL base (its own components, so the per-scene character-LoRA + IP-attn state never tangles
        with the shared keyframe pipe), plus the InstantID image-projection (Resampler) and the IP
        cross-attention processors that inject the insightface face embedding. The few-step distill
        adapter is attached the same way as the keyframe pipe so single-char drafts stay cheap.
        keyframe._render_instantid drives it. Validated on a pod (the attn-processor wiring + the
        ip-adapter.bin key mapping are the parts to eyeball)."""
        if "instantid" in self._cache:
            return self._cache["instantid"]
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from huggingface_hub import hf_hub_download
        from . import instantid as _iid

        base = self.specs[ModelRole.KEYFRAME_BASE]
        id_spec = self.specs[ModelRole.INSTANTID]
        controlnet = ControlNetModel.from_pretrained(
            id_spec.repo_id, subfolder="ControlNetModel", torch_dtype=torch.bfloat16)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base.repo_id, controlnet=controlnet, torch_dtype=torch.bfloat16)
        pipe.to("cuda")
        _set_attention(pipe, self.device)
        pipe._vj_default_scheduler = pipe.scheduler
        self._attach_keyframe_distill(pipe)

        # Identity injection: project the face embedding (Resampler) and wire the IP cross-attention.
        ckpt = torch.load(hf_hub_download(id_spec.repo_id, "ip-adapter.bin"), map_location="cpu")
        pipe._vj_image_proj = _iid.build_image_proj(ckpt["image_proj"]).to("cuda", torch.bfloat16)
        pipe._vj_id_attn = _iid.set_instantid_ip_attn(pipe.unet, ckpt["ip_adapter"])
        self._cache["instantid"] = pipe
        return pipe

    def _attach_keyframe_distill(self, pipe):
        """Load the Hyper-SD few-step distill LoRA as a persistent base adapter "distill" on a
        keyframe-style SDXL pipe (shared by keyframe_pipeline and instantid_pipeline). A finicky
        load degrades to full-step rather than crashing the worker."""
        pipe._vj_distill = None
        few = self.specs.get(ModelRole.KEYFRAME_FEWSTEP)
        if few is None:
            return
        try:
            pipe.load_lora_weights(few.repo_id, weight_name=few.weight_name, adapter_name="distill")
            pipe._vj_distill = "distill"
        except Exception as e:  # noqa: BLE001 -- never let a finicky distill load down the worker
            print(f"keyframe distill LoRA load failed ({few.repo_id} / {few.weight_name}): {e}; "
                  "full-step keyframes only.")

    def i2v_pipeline(self):
        """Wan 2.2 image-to-video: load bf16, quantize the MoE transformers to fp8 with torchao on
        the cards that support it, attach the Lightning distill LoRA for few-step sampling, set the
        attention backend. Final-tier rendering toggles the distill LoRA off (handled by i2v.py);
        here we load the warm baseline."""
        if "i2v" in self._cache:
            return self._cache["i2v"]
        import os
        import torch
        from diffusers import WanImageToVideoPipeline

        spec = self.specs[ModelRole.I2V]
        # The Wan repo ships only bf16 weights (no fp8 variant), so load bf16 and quantize to fp8
        # in place with torchao -- the same path as the SDXL keyframe, not a from_pretrained
        # variant. Wan 2.2 A14B is a ~28B two-expert MoE, too large to hold resident on the tighter
        # cards (and an 80GB H100, where it OOMs), so CPU-offload the inactive expert below the
        # big-VRAM tiers; the H200/B200 keep it resident and quantize to fp8 for full speed.
        pipe = WanImageToVideoPipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16)
        offload = bool(self.device.vram_gb) and self.device.vram_gb < 120
        if not offload:
            pipe.to("cuda")

        # The distill LoRA must be FUSED before fp8 quant: a LoRA cannot load onto torchao-quantized
        # linears (#12535 -> TorchaoLoraLinear), so bake it into the base weights here and then
        # quantize the plain fused model. The few-step distill is the draft/standard speed path; a
        # final-tier worker sets VJ_I2V_DISTILL=0 to keep full steps.
        if os.environ.get("VJ_I2V_DISTILL", "1") != "0":
            distill = self.specs[ModelRole.I2V_DISTILL]
            try:
                pipe.load_lora_weights(distill.repo_id, adapter_name="distill")
                pipe.fuse_lora()
                pipe.unload_lora_weights()
            except Exception as e:  # noqa: BLE001
                print(f"i2v distill LoRA load/fuse failed ({e}); full-step. Fallback: LightX2V loader.")

        use_fp8 = (self.device.supports_fp8 and not offload
                   and os.environ.get("VJ_I2V_FP8", "1") != "0")
        if use_fp8:
            for name in ("transformer", "transformer_2"):  # both MoE experts, after the distill fuse
                module = getattr(pipe, name, None)
                if module is not None:
                    try:
                        _quantize_fp8(module)
                    except Exception as e:  # noqa: BLE001
                        print(f"i2v fp8 quantize of {name} failed ({e}); leaving it bf16.")
        _set_attention(pipe, self.device)
        if offload:
            pipe.enable_model_cpu_offload()  # keep only the active expert on the GPU
        self._cache["i2v"] = pipe
        return pipe

    def unload(self) -> None:
        self._cache.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _quantize_fp8(module) -> None:
    """Quantize a diffusers module to fp8 in place via torchao. Blackwell gets MXFP8, Hopper
    plain fp8; torchao selects the kernel from the device. Verified against the diffusers
    torchao quantization guide."""
    from torchao.quantization import quantize_  # deferred
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

    quantize_(module, Float8DynamicActivationFloat8WeightConfig())


def _set_attention(pipe, device: Device) -> None:
    """Select the attention backend the card supports (FlashAttention-3 on Hopper/Blackwell)."""
    from .device import Attention
    if device.attention() is Attention.FLASH3:
        try:
            pipe.set_attention_backend("flash_attention_3")  # diffusers attention dispatch
        except Exception:
            pass  # fall through to the pipeline default (SDPA)
