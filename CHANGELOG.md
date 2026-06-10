# Changelog

Notable changes per release. Releases are tagged `backend-vX.Y.Z` (SemVer-style,
pre-1.0: PATCH for fixes and backend-only tweaks, MINOR for new features). Entries are
newest-first. History before this file was introduced lives in the git tags; the recent
releases are summarized below from that history.

## Unreleased

**InstantID single-character face identity (the consistent-identity lever).** Wires the
scaffolded-but-dead InstantID path into the GPU keyframe stage: for a single-character shot with
`identity_method="instantid"` and a reference face, the keyframe now uses insightface (antelopev2)
to extract a face embedding + 5 keypoints, projects the embedding to identity tokens through a
Resampler image-projection, injects them via IP cross-attention processors, and conditions a
second (InstantID) ControlNet on the keypoints to pin face structure. Built as its own
`instantid_pipeline()` on a fresh SDXL base (no entanglement with the shared keyframe pipe's
per-scene LoRA / IP-Adapter state), sharing the few-step distill attach so single-char drafts stay
cheap. Multi-character shots and the default IP-Adapter single path are untouched. The face-keypoint
geometry and face selection are pure + unit-tested; the model construction, attn-processor wiring,
and per-render embed-concat defer their imports and are GPU-validation-pending (the parts to eyeball:
the ip-adapter.bin key mapping, the identity-token concat, and the insightface antelopev2 path).
Clean-room: built from the published InstantID architecture + diffusers interfaces, no prior pipeline.

Code: new `instantid.py` (Resampler image-proj, IP attn processor, kps drawing, face analysis),
`models.py` (`face_analyzer`, `instantid_pipeline`, `_attach_keyframe_distill` shared with
`keyframe_pipeline`), `keyframe.py` (`_render_instantid` + the single-char InstantID branch),
`pipeline.py` (thread `identity_method` + instantid scales), `KeyframeParams` fields,
`deploy/requirements.txt` (insightface + onnxruntime-gpu), `tests/test_instantid.py` +
`tests/test_pipeline.py`. CPU suite green.

## backend-v0.1.16

**Wire the few-step keyframe distill into the GPU path (the speed lever).** The tier config
already asked draft/standard for a 4/8-step, cfg=0, DDIM-trailing keyframe, but the pipeline
never loaded the Hyper-SD distill LoRA, never set the scheduler, and the `few_step` flag was
dead, so draft previews ran few-step *without* the LoRA that makes few-step work (degraded
stills). Now the ModelServer loads the Hyper-SDXL distill LoRA as a persistent base adapter
("distill"), `keyframe._bind_loras` gates its weight on the tier (1.0 on draft/standard, 0.0
on final, so one warm pipe serves every tier with no reload), and `keyframe._apply_scheduler`
pins DDIM-trailing for the few-step path and restores a full-step solver for final. The final
tier is untouched; this speeds up previews (the most-repeated op). Validated green on a pod
(2026-06-10): a draft keyframes-only render of all 10 neon_halflife shots came out sharp at
4-step, including the multi-character frames; the feared "8-step LoRA at 4-step draft = soft"
did not happen, so it ships as-is.

Code: `models.py` (ModelSpec.weight_name; load the distill LoRA in `keyframe_pipeline`),
`keyframe.py` (DISTILL_ADAPTER, `KeyframeParams.scheduler`, `_apply_scheduler`, few-step gating
in `_bind_loras`), `pipeline.py` (thread `scheduler` through `keyframe_params_from`),
`tests/test_keyframe.py` + `tests/test_pipeline.py` (distill-weight + scheduler mapping).
Full suite green.

Public-readiness housekeeping ahead of open-sourcing the repository: added
`CONTRIBUTING.md` (clean-room / independence-protective posture, house rules, DCO),
`SECURITY.md` (private vulnerability reporting + the render-backend security boundary),
`CODE_OF_CONDUCT.md`, and this `CHANGELOG.md`. Corrected the README architecture table to
reflect the shipped pipeline, and removed em-dashes from the README to match the house
style.

## backend-v0.1.15

Multi-character pose: wire the OpenPose ControlNet so a 2+ character shot plants two
distinct bodies instead of a blended one (`keyframe.py`).

## backend-v0.1.14

Stamp `user_email` on uploaded artifacts so the control plane's `/api/artifact` ownership
check can serve them back (`harness`).

## backend-v0.1.13

Fail loud when a staged LoRA registers no adapter, and make reused-LoRA injection
fail-fast too, so a silent no-op never ships a render without the intended character
(`keyframe.py`, `pipeline.py`).

## backend-v0.1.12

First-class `preview` action for keyframes-only renders: the orchestrator short-circuits
after the SDXL pass when only keyframes are requested (`orchestrator.py`).

## backend-v0.1.11

Cache both Wan 2.2 MoE experts to clear the step-12 hit-cliff in image-to-video
(`i2v.py`).

## backend-v0.1.10

Use the matched `enable_cache` / `disable_cache` pair for the i2v feature cache (`i2v.py`).

## backend-v0.1.2 -- backend-v0.1.9

Earlier pipeline build-out: the structured render progress channel (R2-backed), the
feature-cache denoise accelerator, reused-LoRA staging from R2 so warm workers skip
retraining, the R2-mirror model loader, and the AGPL-3.0-only license. See the git tags
for the per-release detail.
