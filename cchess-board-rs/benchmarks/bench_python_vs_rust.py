"""Performance benchmark: Python (OpenCV) vs Rust (PyO3 bindings).

Compares equivalent operations between the original Python implementation
(using OpenCV) and the new pure-Rust implementation (via PyO3 bindings).

Run:
    python benchmarks/bench_python_vs_rust.py
    python benchmarks/bench_python_vs_rust.py --json   # JSON output
    python benchmarks/bench_python_vs_rust.py --iterations 500
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List

# Rust (PyO3 binding) implementation
import cchess_board_rs
import cv2
import numpy as np

# Python (OpenCV-based) implementation
from cchess_board.detector import (
    check_keypoints as py_check_keypoints,
)
from cchess_board.detector import (
    corner_points_to_rect as py_corner_points_to_rect,
)
from cchess_board.detector import (
    extract_chessboard as py_extract_chessboard,
)
from cchess_board.detector import (
    get_board_corner_points as py_get_board_corner_points,
)
from cchess_board.detector import (
    perspective_transform as py_perspective_transform,
)
from cchess_board_rs import Image as RsImage
from cchess_board_rs import (
    py_check_keypoints as rs_check_keypoints,
)
from cchess_board_rs import (
    py_corner_points_to_rect as rs_corner_points_to_rect,
)
from cchess_board_rs import (
    py_extract_chessboard as rs_extract_chessboard,
)
from cchess_board_rs import (
    py_get_board_corner_points as rs_get_board_corner_points,
)
from cchess_board_rs import (
    py_get_perspective_transform as rs_get_perspective_transform,
)
from cchess_board_rs import (
    py_invert_affine_2x3 as rs_invert_affine_2x3,
)
from cchess_board_rs import (
    py_invert_perspective_3x3 as rs_invert_perspective_3x3,
)
from cchess_board_rs import (
    py_perspective_transform as rs_perspective_transform,
)
from cchess_board_rs import (
    py_perspective_transform_points as rs_perspective_transform_points,
)
from cchess_board_rs import (
    py_warp_affine as rs_warp_affine,
)
from cchess_board_rs import (
    py_warp_perspective as rs_warp_perspective,
)


@dataclass
class BenchResult:
    name: str
    py_ms: float
    rs_ms: float
    iterations: int
    speedup: float = field(init=False)

    def __post_init__(self):
        self.speedup = self.py_ms / self.rs_ms if self.rs_ms > 0 else float("inf")


def time_op(fn: Callable, iterations: int) -> float:
    """Return median time in milliseconds over `iterations` runs."""
    times: List[float] = []
    # Warm-up
    fn()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


# ─── Helpers ────────────────────────────────────────────────────────────────


def make_keypoints() -> np.ndarray:
    """Generate realistic-looking 4 keypoints (A0, A8, J0, J8)."""
    return np.array(
        [
            [85.0, 95.0],  # A0 (top-left)
            [415.0, 90.0],  # A8 (top-right)
            [90.0, 405.0],  # J0 (bottom-left)
            [420.0, 395.0],  # J8 (bottom-right)
        ],
        dtype=np.float32,
    )


def make_image_cv2(width: int, height: int) -> cv2.UMat:
    """Create a BGR image as cv2.UMat (matches Python detector API)."""
    np_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return cv2.UMat(np_img)


def make_image_rs(width: int, height: int) -> RsImage:
    """Create an equivalent image for the Rust binding."""
    np_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return RsImage.from_array(np_img)


# ─── Benchmarks ─────────────────────────────────────────────────────────────


def bench_check_keypoints(iterations: int) -> BenchResult:
    kps = make_keypoints()

    def py_run():
        py_check_keypoints(kps)

    def rs_run():
        rs_check_keypoints(kps.tolist())

    return BenchResult(
        "check_keypoints (4-point validation)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_get_board_corner_points(iterations: int) -> BenchResult:
    kps = make_keypoints()

    def py_run():
        py_get_board_corner_points(kps)

    def rs_run():
        rs_get_board_corner_points(kps.tolist())

    return BenchResult(
        "get_board_corner_points (min/max bounding box)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_corner_points_to_rect(iterations: int) -> BenchResult:
    corners = make_keypoints()

    def py_run():
        py_corner_points_to_rect(corners)

    def rs_run():
        rs_corner_points_to_rect(corners.tolist())

    return BenchResult(
        "corner_points_to_rect (rect with half-grid padding)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_get_perspective_transform(iterations: int) -> BenchResult:
    src = make_keypoints().astype(np.float32)
    dst = np.array(
        [
            [50.0, 50.0],
            [400.0, 50.0],
            [50.0, 450.0],
            [400.0, 450.0],
        ],
        dtype=np.float32,
    )

    def py_run():
        cv2.getPerspectiveTransform(src, dst)

    def rs_run():
        rs_get_perspective_transform(src.tolist(), dst.tolist())

    return BenchResult(
        "get_perspective_transform (homography from 4 pairs)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_perspective_transform_points(iterations: int) -> BenchResult:
    src = make_keypoints().astype(np.float32)
    dst = np.array(
        [
            [50.0, 50.0],
            [400.0, 50.0],
            [50.0, 450.0],
            [400.0, 450.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)

    def py_run():
        pts = src.reshape(-1, 1, 2).astype(np.float32)
        cv2.perspectiveTransform(pts, matrix)

    def rs_run():
        rs_perspective_transform_points(src.tolist(), matrix.tolist())

    return BenchResult(
        "perspective_transform_points (apply matrix to 4 points)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_warp_perspective(iterations: int) -> BenchResult:
    width, height = 500, 500
    img_py = make_image_cv2(width, height)
    img_rs = make_image_rs(width, height)
    src = np.array(
        [
            [50.0, 50.0],
            [450.0, 50.0],
            [50.0, 450.0],
            [450.0, 450.0],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [50.0, 50.0],
            [400.0, 50.0],
            [50.0, 450.0],
            [400.0, 450.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)

    def py_run():
        cv2.warpPerspective(img_py, matrix, (450, 500))

    def rs_run():
        rs_warp_perspective(img_rs, matrix.tolist(), 450, 500)

    return BenchResult(
        f"warp_perspective (500x500 -> 450x500)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_warp_perspective_large(iterations: int) -> BenchResult:
    """Larger image to stress the bilinear interpolation loop."""
    # 1280x1280 bilinear is ~340ms/iter in Rust vs ~5ms in OpenCV,
    # so use a much smaller iteration count by default.
    width, height = 1280, 1280
    img_py = make_image_cv2(width, height)
    img_rs = make_image_rs(width, height)
    src = np.array(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(height)],
            [float(width), float(height)],
        ],
        dtype=np.float32,
    )
    dst = src.copy()
    matrix = cv2.getPerspectiveTransform(src, dst)

    def py_run():
        cv2.warpPerspective(img_py, matrix, (width, height))

    def rs_run():
        rs_warp_perspective(img_rs, matrix.tolist(), width, height)

    return BenchResult(
        f"warp_perspective (1280x1280 identity)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_warp_affine(iterations: int) -> BenchResult:
    width, height = 500, 500
    img_py = make_image_cv2(width, height)
    img_rs = make_image_rs(width, height)
    matrix = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]], dtype=np.float32)

    def py_run():
        cv2.warpAffine(img_py, matrix, (width, height))

    def rs_run():
        rs_warp_affine(img_rs, matrix.tolist(), width, height)

    return BenchResult(
        f"warp_affine (500x500 with translation)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_invert_perspective(iterations: int) -> BenchResult:
    src = make_keypoints().astype(np.float32)
    dst = np.array(
        [
            [50.0, 50.0],
            [400.0, 50.0],
            [50.0, 450.0],
            [400.0, 450.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)

    def py_run():
        np.linalg.inv(matrix)

    def rs_run():
        rs_invert_perspective_3x3(matrix.tolist())

    return BenchResult(
        "invert_perspective_3x3 (Gauss-Jordan vs numpy.linalg.inv)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_invert_affine(iterations: int) -> BenchResult:
    matrix = np.array([[1.5, 0.3, 10.0], [-0.2, 1.2, -5.0]], dtype=np.float32)
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])

    def py_run():
        np.linalg.inv(full)[:2]

    def rs_run():
        rs_invert_affine_2x3(matrix.tolist())

    return BenchResult(
        "invert_affine_2x3",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_color_conversion(iterations: int) -> BenchResult:
    """BGR <-> RGB conversion (numpy slice vs Rust method)."""
    np_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

    def py_run():
        # Standard OpenCV/idiom conversion: reverse channel order
        np_img[..., ::-1].copy()

    def rs_run():
        img = RsImage.from_array(np_img.copy())
        img.bgr_to_rgb()

    return BenchResult(
        "color conversion (500x500 BGR<->RGB, copy)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_crop(iterations: int) -> BenchResult:
    """Crop a 500x500 region."""
    np_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

    def py_run():
        np_img[100:400, 100:400].copy()

    def rs_run():
        img = RsImage.from_array(np_img.copy())
        img.crop(100, 100, 300, 300)

    return BenchResult(
        "crop (500x500 -> 300x300 region)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_resize(iterations: int) -> BenchResult:
    """Resize a 1000x1000 image down to 500x500."""
    np_img = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)

    def py_run():
        cv2.resize(np_img, (500, 500), interpolation=cv2.INTER_LINEAR)

    def rs_run():
        img = RsImage.from_array(np_img.copy())
        img.resize(500, 500)

    return BenchResult(
        "resize (1000x1000 -> 500x500 bilinear)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


def bench_extract_chessboard(iterations: int) -> BenchResult:
    """Full pipeline: corners -> warp -> return rectified image."""
    width, height = 500, 500
    img_py = make_image_cv2(width, height)
    img_rs = make_image_rs(width, height)
    kps = make_keypoints()

    def py_run():
        py_extract_chessboard(img_py, kps)

    def rs_run():
        rs_extract_chessboard(img_rs, kps.tolist())

    return BenchResult(
        "extract_chessboard (full rectify pipeline)",
        time_op(py_run, iterations),
        time_op(rs_run, iterations),
        iterations,
    )


# ─── Reporting ──────────────────────────────────────────────────────────────


def print_table(results: List[BenchResult]):
    name_w = max(len(r.name) for r in results) + 2
    py_w = 12
    rs_w = 12
    speed_w = 10

    header = (
        f"{'Operation':<{name_w}} "
        f"{'Python (ms)':>{py_w}} "
        f"{'Rust (ms)':>{rs_w}} "
        f"{'Speedup':>{speed_w}}"
    )
    print("=" * len(header))
    print("Python (OpenCV) vs Rust (PyO3) — median time over N iterations")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        speedup_str = (
            f"{r.speedup:>{speed_w - 2}.2f}x"
            if r.speedup < 1000
            else f"{'very fast':>{speed_w}}"
        )
        print(
            f"{r.name:<{name_w}} "
            f"{r.py_ms:>{py_w}.4f} "
            f"{r.rs_ms:>{rs_w}.4f} "
            f"{speedup_str}"
        )
    print("=" * len(header))


def run_all(iterations: int) -> List[BenchResult]:
    # Light operations: use the requested iteration count
    light_benches = [
        bench_check_keypoints,
        bench_get_board_corner_points,
        bench_corner_points_to_rect,
        bench_get_perspective_transform,
        bench_perspective_transform_points,
        bench_invert_perspective,
        bench_invert_affine,
        bench_color_conversion,
        bench_crop,
        bench_resize,
    ]
    # Heavy operations (bilinear warp is O(N) with a per-pixel loop)
    # Use a smaller iteration count so the script completes in reasonable time.
    heavy_iters = max(5, iterations // 40)
    heavy_benches = [
        bench_warp_affine,
        bench_warp_perspective,
        bench_warp_perspective_large,
        bench_extract_chessboard,
    ]

    results = []
    for fn in light_benches:
        try:
            r = fn(iterations)
            results.append(r)
        except Exception as e:
            print(f"[!] {fn.__name__} failed: {e}", file=sys.stderr)
    for fn in heavy_benches:
        try:
            r = fn(heavy_iters)
            results.append(r)
        except Exception as e:
            print(f"[!] {fn.__name__} failed: {e}", file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python (OpenCV) vs Rust (PyO3) cchess implementations"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=200, help="Iterations per benchmark"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of table"
    )
    args = parser.parse_args()

    if args.json:
        # JSON output: shorter warm-up iterations, then dump
        results = run_all(args.iterations)
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        results = run_all(args.iterations)
        print()
        print_table(results)
        print()
        total_py = sum(r.py_ms for r in results)
        total_rs = sum(r.rs_ms for r in results)
        if total_rs > 0:
            print(f"Total Python time: {total_py:.4f} ms")
            print(f"Total Rust time:   {total_rs:.4f} ms")
            print(f"Aggregate speedup: {total_py / total_rs:.2f}x")


if __name__ == "__main__":
    main()
