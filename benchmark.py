import time
import subprocess
import sys
import os

def bench_python(iterations=50):
    """Benchmark using the Python package (which calls Rust via FFI)."""
    try:
        import cchess_board
    except Exception as e:
        print(f"Failed to import cchess_board: {e}")
        return None
    # Prepare detector and classifier
    det = cchess_board.Detector()
    clf = cchess_board.Classifier()
    # Load test image
    img_path = os.path.join('samples', 'board1.jpg')
    if not os.path.exists(img_path):
        print(f"Test image not found: {img_path}")
        return None
    # Read image as bytes (assuming the API expects numpy array or path?)
    # We'll assume the detector accepts a file path string.
    # If not, we can adjust later.
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        # Detect board
        board_info = det.detect(img_path)
        # Classify pieces
        pieces = clf.classify(img_path, board_info)
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    return {
        'avg_time_sec': avg_time,
        'fps': fps,
        'iterations': iterations,
        'total_time_sec': sum(times)
    }

def bench_rust(iterations=50):
    """Benchmark using a Rust binary built for benchmarking."""
    # Build the benchmark binary if not present
    bench_dir = os.path.join('cchess-board-rs', 'target', 'release')
    bench_exe = os.path.join(bench_dir, 'bench_detect_classify.exe' if sys.platform.startswith('win') else 'bench_detect_classify')
    if not os.path.exists(bench_exe):
        print("Building Rust benchmark binary...")
        result = subprocess.run(['cargo', 'build', '--release', '--bin', 'bench_detect_classify'],
                                cwd='cchess-board-rs', capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to build benchmark: {result.stderr}")
            return None
    # Run benchmark
    img_path = os.path.join('samples', 'board1.jpg')
    if not os.path.exists(img_path):
        print(f"Test image not found: {img_path}")
        return None
    cmd = [bench_exe, img_path, str(iterations)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd='cchess-board-rs', capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Benchmark failed: {result.stderr}")
        return None
    # Expect output like: avg_time:0.0123,fps:81.3
    out = result.stdout.strip()
    try:
        parts = out.split(',')
        avg_time = float(parts[0].split(':')[1])
        fps = float(parts[1].split(':')[1])
        return {
            'avg_time_sec': avg_time,
            'fps': fps,
            'iterations': iterations,
            'total_time_sec': avg_time * iterations
        }
    except Exception as e:
        print(f"Failed to parse benchmark output: {out} ({e})")
        return None

def main():
    print("=== Performance Benchmark: Rust vs Python (FFI) ===")
    py_res = bench_python()
    rs_res = bench_rust()
    if py_res:
        print("\nPython package (calls Rust via FFI):")
        print(f"  Average time per run: {py_res['avg_time_sec']*1000:.2f} ms")
        print(f"  FPS: {py_res['fps']:.2f}")
    if rs_res:
        print("\nPure Rust binary:")
        print(f"  Average time per run: {rs_res['avg_time_sec']*1000:.2f} ms")
        print(f"  FPS: {rs_res['fps']:.2f}")
    if py_res and rs_res:
        speedup = py_res['avg_time_sec'] / rs_res['avg_time_sec'] if rs_res['avg_time_sec'] > 0 else 0
        print(f"\nSpeedup (Python/FFI over Rust): {speedup:.2f}x")

if __name__ == '__main__':
    main()