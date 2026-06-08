# vivijure-backend

A clean-room render backend for Vivijure: it consumes a project bundle (a standard
`storyboard.yaml` plus the cast), generates SDXL keyframes, trains per-character LoRAs,
turns the keyframes into motion with image-to-video, and returns the artifacts in the
shape the Vivijure control plane already expects.

## Why this exists

This is an independent reimplementation, written from the control-plane API contract
and the underlying models' own documentation. It deliberately shares no source with any
prior render pipeline; the contract (the `storyboard.yaml` schema, the cast registry, the
render-job input/output) is the only thing carried over, and that contract is the control
plane's own.

## The contract (what the backend has to satisfy)

Input: a bundle (`.tar.gz`) the control plane writes to object storage, containing
- `storyboard.yaml` — the validated storyboard (see `contract.py:Storyboard`): a title,
  style fields, `use_characters`, and a list of scenes, each with a `prompt`, optional
  `character_slots`, beat timing (`start` / `end` / `target_seconds`), and an optional
  `start_image`.
- `characters/registry.json` — slot to `{name, prompt, image}`.
- `characters/refs/<SLOT>/ref_NN.<ext>` — per-character training / IP-Adapter references.

Output: the render-job result (see `contract.py:RenderResult`): the final `output_key`,
the per-shot `keyframes` and `clips`, the trained-LoRA ids, and a `state` key, all in the
keys/shape the control plane polls for.

## Architecture (build order)

| Module | Role | Status |
|---|---|---|
| `contract.py` | storyboard.yaml + cast + job I/O types; bundle reader | scaffolding |
| `models.py` | load SDXL, InstantID, IP-Adapter, OpenPose ControlNet, Wan i2v | todo |
| `lora_train.py` | SDXL character LoRA from refs | todo |
| `keyframe.py` | per-scene SDXL keyframe; regional + pose for 2+ char shots | todo |
| `i2v.py` | Wan image-to-video, keyframe to clip | todo |
| `assemble.py` | concat / off-GPU finish | todo |
| `orchestrator.py` | train, render, export; the render-job entrypoint | todo |

The serverless harness (RunPod handler, object-store I/O, model mirroring) plugs in on
top; the pipeline modules above expose a clean interface it drives.

## License

TBD (owner's choice). Nothing here is encumbered by third-party pipeline code.
