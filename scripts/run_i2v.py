#!/usr/bin/env python3
"""Pod driver: animate a keyframe into its shot's clip via Wan 2.2 i2v.

    python scripts/run_i2v.py BUNDLE OUT_DIR --shot shot_01 --keyframe /path/shot_01.png \
        [--quality draft|standard|final]

Run on a CUDA pod (Hopper/Blackwell for the fp8 path) with diffusers/transformers installed.
The keyframe is the first frame; the scene prompt is the motion. Validates i2v.animate end to end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vivijure_backend import i2v  # noqa: E402
from vivijure_backend.contract import Bundle  # noqa: E402
from vivijure_backend.keyframe import build_prompt  # noqa: E402
from vivijure_backend.models import ModelServer  # noqa: E402
from vivijure_backend.routing import QualityTier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--keyframe", type=Path, required=True)
    ap.add_argument("--quality", default="draft")
    ap.add_argument("--height", type=int, help="override render height (else follows keyframe)")
    ap.add_argument("--width", type=int, help="override render width")
    args = ap.parse_args()

    bundle = Bundle.extract(args.bundle, args.out_dir / "bundle")
    scene = {s.id: s for s in bundle.storyboard.scenes}[args.shot]
    prompt = build_prompt(scene, bundle.cast, bundle.storyboard)
    params = i2v.params_for(scene, QualityTier.parse(args.quality))
    if args.height:
        params.height = args.height
    if args.width:
        params.width = args.width
    print(f"shot {args.shot}: {params.num_frames} frames @ {params.fps}fps, "
          f"{'distilled ' + str(params.steps) + '-step' if params.distill else 'full-step'}", flush=True)

    server = ModelServer()
    t = time.time()
    res = i2v.animate(scene, args.keyframe, prompt, server, args.out_dir / f"{args.shot}.mp4", params=params)
    print(f"animated in {time.time() - t:.0f}s", flush=True)
    print(json.dumps({
        "shot": res.shot_id, "frames": res.num_frames, "fps": res.fps,
        "seconds": res.seconds, "distilled": res.distilled,
        "path": str(res.path), "exists": res.path.is_file(),
    }, indent=2))
    return 0 if res.path.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
