#!/usr/bin/env python3
"""Pod driver: render one scene's keyframe from a bundle.

    python scripts/run_keyframe.py BUNDLE OUT_DIR --shot shot_02 \
        --lora A=/path/A.safetensors --lora B=/path/B.safetensors [--steps 30 --full-step --seed 0]

Run on a CUDA pod with diffusers/transformers/peft installed. Prints the scene roster (so the
single-vs-regional path per shot is visible), then renders the requested shot. The harness calls
keyframe.render_keyframe directly; this is the standalone build-and-run driver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vivijure_backend.contract import Bundle  # noqa: E402
from vivijure_backend.keyframe import KeyframeParams, engine_for, render_keyframe  # noqa: E402
from vivijure_backend.models import ModelServer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--shot", help="shot id to render; default = first scene")
    ap.add_argument("--lora", action="append", default=[], help="SLOT=path, repeatable")
    ap.add_argument("--steps", type=int, default=KeyframeParams.steps)
    ap.add_argument("--full-step", action="store_true", help="plain SDXL (no few-step distill)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bundle = Bundle.extract(args.bundle, args.out_dir / "bundle")
    params = KeyframeParams(steps=args.steps, few_step=not args.full_step, seed=args.seed)
    print("scene roster:")
    for sc in bundle.storyboard.scenes:
        print(f"  {sc.id}: slots={sc.character_slots} -> {engine_for(sc, params)}")

    loras = {}
    for spec in args.lora:
        slot, path = spec.split("=", 1)
        loras[slot] = Path(path)

    scenes = {sc.id: sc for sc in bundle.storyboard.scenes}
    shot = args.shot or bundle.storyboard.scenes[0].id
    if shot not in scenes:
        print(f"no such shot: {shot}", file=sys.stderr)
        return 1

    server = ModelServer()
    res = render_keyframe(scenes[shot], bundle.cast, bundle.storyboard, server,
                          args.out_dir / f"{shot}.png", params=params, lora_paths=loras)
    print(json.dumps({
        "shot": res.shot_id, "engine": res.engine, "multi_char": res.multi_char,
        "slots": res.slots, "seed": res.seed, "prompt": res.prompt,
        "path": str(res.path), "exists": res.path.is_file(),
    }, indent=2))
    return 0 if res.path.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
