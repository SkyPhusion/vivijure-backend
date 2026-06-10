"""CPU tests for the pure InstantID helpers: face selection and keypoint drawing. The GPU bodies
(image-projection, attn-processor wiring, the per-render call) defer torch/diffusers/insightface
and are validated on a pod, not here."""
from dataclasses import dataclass

from vivijure_backend.instantid import draw_kps, largest_face, KPS_COLORS


@dataclass
class _Face:
    bbox: tuple


def test_largest_face_picks_the_biggest_bbox():
    small = _Face((0, 0, 10, 10))      # area 100
    big = _Face((0, 0, 100, 120))      # area 12000
    mid = _Face((0, 0, 50, 50))        # area 2500
    assert largest_face([small, big, mid]) is big


def test_largest_face_empty_is_none():
    assert largest_face([]) is None
    assert largest_face(None) is None


def test_draw_kps_canvas_size_and_black_background():
    kps = [(20, 20), (80, 20), (50, 50), (30, 80), (70, 80)]
    img = draw_kps(128, 128, kps)
    assert img.size == (128, 128)
    assert img.getpixel((0, 0)) == (0, 0, 0)        # corner stays black (no landmark there)


def test_draw_kps_marks_each_landmark():
    kps = [(20, 20), (80, 20), (50, 50), (30, 80), (70, 80)]
    img = draw_kps(128, 128, kps)
    # Each landmark center is painted its fixed InstantID colour (not black).
    for (x, y) in kps:
        assert img.getpixel((x, y)) != (0, 0, 0)


def test_draw_kps_handles_fewer_than_five_points():
    img = draw_kps(64, 64, [(10, 10), (30, 30)])     # 2 points: no crash, still a 64x64 canvas
    assert img.size == (64, 64)
    assert len(KPS_COLORS) == 5
