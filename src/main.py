"""
FrameForge — main.py
=====================
CLI entry point that orchestrates the full performance analysis pipeline.

Usage:
    python main.py --full              Run full pipeline
    python main.py --analyze           Analyze FrameView logs only
    python main.py --cuda              Analyze CUDA benchmark results only
    python main.py --dashboard         Regenerate dashboards from existing data
    python main.py --compare a.csv b.csv   Compare two FrameView captures
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from frameview_parser import FrameViewParser
from regression_detector import RegressionDetector
from dashboard import DashboardGenerator


# =============================================================================
# Config
# =============================================================================
FRAMEVIEW_DIR = Path(__file__).parent.parent / "frameview_logs"
RESULTS_DIR = Path(__file__).parent.parent / "results"
NSIGHT_DIR = Path(__file__).parent.parent / "nsight_profiles"
CUDA_RESULTS = RESULTS_DIR / "cuda_results.json"
CUDA_BINARY = Path(__file__).parent / "cuda_bench"


def print_header():
    print()
    print("=" * 60)
    print("  🎮 FrameForge — GPU Gaming Performance Pipeline")
    print("=" * 60)
    print()


# =============================================================================
# Pipeline Steps
# =============================================================================
def run_cuda_benchmark():
    """Compile and run the C++ CUDA benchmark."""
    print("[CUDA] Running C++ CUDA shader benchmarks...")

    # Check if binary exists
    binary = CUDA_BINARY
    if sys.platform == "win32":
        binary = binary.with_suffix(".exe")

    if not binary.exists():
        print(f"  Binary not found: {binary}")
        print(f"  Compile first: cd src && make")

        # Try to compile
        makefile = Path(__file__).parent / "Makefile"
        if makefile.exists():
            print("  Attempting to compile...")
            result = subprocess.run(["make", "-C", str(Path(__file__).parent)],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Compilation failed: {result.stderr}")
                return None
        else:
            return None

    # Run benchmark
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [str(binary)],
            capture_output=True, text=True, timeout=300
        )
        # stdout = JSON, stderr = progress
        print(result.stderr)

        if result.stdout.strip():
            with open(CUDA_RESULTS, "w") as f:
                f.write(result.stdout)
            print(f"  ✓ CUDA results saved: {CUDA_RESULTS}")
            return json.loads(result.stdout)
        else:
            print("  ✗ No JSON output from cuda_bench")
            return None

    except subprocess.TimeoutExpired:
        print("  ✗ CUDA benchmark timed out")
        return None
    except FileNotFoundError:
        print(f"  ✗ Binary not found: {binary}")
        return None


def run_nsight_profile():
    """Profile CUDA benchmark with Nsight Systems."""
    print("[NSIGHT] Profiling with Nsight Systems...")

    binary = CUDA_BINARY
    if sys.platform == "win32":
        binary = binary.with_suffix(".exe")

    if not binary.exists():
        print(f"  ✗ CUDA binary not found. Compile first.")
        return

    NSIGHT_DIR.mkdir(parents=True, exist_ok=True)
    output = NSIGHT_DIR / "cuda_bench_profile"

    try:
        result = subprocess.run([
            "nsys", "profile",
            "--trace=cuda,nvtx",
            "--force-overwrite=true",
            f"--output={output}",
            str(binary)
        ], capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"  ✓ Nsight profile saved: {output}.nsys-rep")
            print(f"  Open in Nsight Systems GUI to view timeline.")
        else:
            print(f"  ✗ Nsight profiling failed: {result.stderr[:200]}")

    except FileNotFoundError:
        print("  ✗ nsys not found. Install Nsight Systems:")
        print("    https://developer.nvidia.com/nsight-systems")
    except subprocess.TimeoutExpired:
        print("  ✗ Nsight profiling timed out")


def analyze_frameview_logs():
    """Parse and analyze all FrameView CSV logs."""
    print(f"[FRAMEVIEW] Analyzing logs in {FRAMEVIEW_DIR}/...")

    parser = FrameViewParser()
    results = parser.parse_directory(FRAMEVIEW_DIR)

    if not results:
        print(f"  No CSV files found. To add data:")
        print(f"  1. Download FrameView: https://www.nvidia.com/en-us/geforce/technologies/frameview/")
        print(f"  2. Run FrameView → launch a game → press F10 to capture")
        print(f"  3. Copy the CSV to {FRAMEVIEW_DIR}/")

    return results


def analyze_cuda_results():
    """Analyze existing CUDA benchmark results."""
    print(f"[CUDA] Analyzing results from {CUDA_RESULTS}...")

    if not CUDA_RESULTS.exists():
        print(f"  ✗ No CUDA results found. Run: python main.py --cuda-run")
        return None

    with open(CUDA_RESULTS) as f:
        data = json.load(f)

    print(f"  GPU: {data.get('gpu', 'Unknown')}")
    print(f"  VRAM: {data.get('vram_mb', '?')} MB")
    print(f"  Results: {len(data.get('results', []))} benchmarks")

    for r in data.get("results", []):
        print(f"    [{r['resolution']}] {r['kernel']}: "
              f"{r['avg_ms']:.2f}ms avg | {r['throughput']:.1f} {r['unit']}")

    return data


def detect_regressions(fv_results, cuda_data):
    """Run regression detection on all available data."""
    print("[REGRESSION] Detecting performance regressions...")

    detector = RegressionDetector()
    all_regressions = []

    # CUDA regressions (cross-resolution)
    if cuda_data or CUDA_RESULTS.exists():
        cuda_regs = detector.compare_cuda_results(CUDA_RESULTS)
        all_regressions.extend(cuda_regs)
        if cuda_regs:
            print(f"  CUDA: {len(cuda_regs)} regressions found")
            for r in cuda_regs:
                print(f"    [{r['severity']}] {r['kernel']}: "
                      f"{r['from_res']}→{r['to_res']} ({r['change_pct']:+.1f}%)")
        else:
            print("  CUDA: ✅ No regressions")

    # FrameView regressions (comparing runs)
    if fv_results and len(fv_results) >= 2:
        stats_list = [stats for _, _, stats in fv_results]
        fv_reports = detector.compare_resolutions(stats_list)
        for report in fv_reports:
            all_regressions.extend(report.get("regressions", []))
            if report["regressions"]:
                detector.print_report(report)

    return all_regressions


def generate_dashboards(fv_results, cuda_data, regressions):
    """Generate all performance dashboard charts."""
    print(f"[DASHBOARD] Generating charts in {RESULTS_DIR}/...")

    gen = DashboardGenerator(output_dir=str(RESULTS_DIR))

    # Chart 1: FPS comparison (from FrameView)
    if fv_results:
        stats_list = [stats for _, _, stats in fv_results]
        gen.plot_fps_comparison(stats_list)
        gen.plot_summary_table(stats_list)

        # Chart 2: Frame time distributions
        dfs = [df for _, df, _ in fv_results]
        labels = [name for name, _, _ in fv_results]
        gen.plot_frame_time_distribution(dfs, labels)

        # Chart 3: Stutter timeline (first/largest capture)
        largest = max(fv_results, key=lambda x: len(x[1]))
        gen.plot_frame_time_timeline(largest[1], label=largest[0])

    # Chart 4: CUDA throughput
    if CUDA_RESULTS.exists():
        gen.plot_cuda_results(str(CUDA_RESULTS))

    # Chart 5: Regression report
    gen.plot_regression_report(regressions)


def compare_two_files(file_a, file_b):
    """Compare two FrameView CSV files directly."""
    print(f"[COMPARE] {file_a} vs {file_b}")

    parser = FrameViewParser()
    detector = RegressionDetector()

    df_a = parser.parse(file_a)
    df_b = parser.parse(file_b)
    stats_a = parser.analyze(df_a, label=Path(file_a).stem)
    stats_b = parser.analyze(df_b, label=Path(file_b).stem)

    report = detector.compare(stats_a, stats_b)
    detector.print_report(report)

    # Generate comparison charts
    gen = DashboardGenerator(output_dir=str(RESULTS_DIR))
    gen.plot_fps_comparison([stats_a, stats_b])
    gen.plot_frame_time_distribution([df_a, df_b], [stats_a["label"], stats_b["label"]])
    gen.plot_regression_report(report.get("regressions", []),
                               title=f"Regression: {stats_a['label']} → {stats_b['label']}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="FrameForge — GPU Gaming Performance Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full                 Run full pipeline
  python main.py --analyze              Analyze FrameView logs only
  python main.py --cuda-run             Compile + run CUDA benchmarks
  python main.py --cuda                 Analyze existing CUDA results
  python main.py --profile              Profile CUDA benchmark with Nsight
  python main.py --dashboard            Regenerate charts from existing data
  python main.py --compare a.csv b.csv  Compare two captures
        """
    )

    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--analyze", action="store_true", help="Analyze FrameView logs")
    parser.add_argument("--cuda-run", action="store_true", help="Run CUDA benchmarks")
    parser.add_argument("--cuda", action="store_true", help="Analyze CUDA results")
    parser.add_argument("--profile", action="store_true", help="Nsight Systems profiling")
    parser.add_argument("--dashboard", action="store_true", help="Generate dashboards")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
                        help="Compare two FrameView CSV files")

    args = parser.parse_args()

    # Default to --full if no args
    if not any(vars(args).values()):
        args.full = True

    print_header()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fv_results = []
    cuda_data = None
    regressions = []

    if args.compare:
        compare_two_files(args.compare[0], args.compare[1])
        return

    if args.full or args.cuda_run:
        cuda_data = run_cuda_benchmark()

    if args.full or args.profile:
        run_nsight_profile()

    if args.full or args.analyze:
        fv_results = analyze_frameview_logs()

    if args.full or args.cuda:
        if cuda_data is None:
            cuda_data = analyze_cuda_results()

    if args.full:
        regressions = detect_regressions(fv_results, cuda_data)

    if args.full or args.dashboard:
        if not fv_results:
            fv_results = analyze_frameview_logs()
        generate_dashboards(fv_results, cuda_data, regressions)

    # Final summary
    print()
    print("=" * 60)
    print("  📊 Pipeline Complete")
    print("=" * 60)
    n_charts = len(list(RESULTS_DIR.glob("*.png")))
    print(f"  Charts generated: {n_charts}")
    print(f"  Results dir:      {RESULTS_DIR.resolve()}")
    if CUDA_RESULTS.exists():
        print(f"  CUDA results:     {CUDA_RESULTS}")
    nsight_files = list(NSIGHT_DIR.glob("*.nsys-rep"))
    if nsight_files:
        print(f"  Nsight profiles:  {len(nsight_files)} file(s)")
    print()


if __name__ == "__main__":
    main()