"""
FrameForge — frameview_parser.py
=================================
Parses NVIDIA FrameView and PresentMon CSV logs from real game benchmarks.
Computes FPS percentiles, frame time statistics, stutter detection, and
smoothness scoring — the same analysis NVIDIA's gaming performance team does.

Supports: DirectX 9/11/12, Vulkan, OpenGL (all captured by FrameView)

Usage:
    from frameview_parser import FrameViewParser
    parser = FrameViewParser()
    df = parser.parse("frameview_logs/MyGame.csv")
    stats = parser.analyze(df, label="MyGame 1080p Ultra")
"""

import pandas as pd
import numpy as np
from pathlib import Path


class FrameViewParser:
    """Parses and analyzes FrameView / PresentMon CSV frame data."""

    # Possible frame time column names across FrameView versions and PresentMon
    FRAME_TIME_COLS = [
        "MsBetweenPresents",        # FrameView / PresentMon standard
        "msBetweenPresents",        # lowercase variant
        "MsBetweenDisplayChange",   # displayed frame time
        "frametime",                # generic
        "FrameTime",                # generic
        "frame_time_ms",            # custom
    ]

    # Metadata columns to extract if available
    META_COLS = [
        "Application", "Runtime", "Resolution",
        "GPU", "GPU0Name", "CPU",
        "MsUntilRenderComplete",    # render latency
        "MsUntilDisplayed",         # display latency
        "MsInPresentAPI",           # API overhead
        "GPUPower(W)",              # power consumption
        "GPUTemperature(C)",        # temperature
        "GPUUtilization(%)",        # utilization
        "GPUFrequency(MHz)",        # clock speed
    ]

    def parse(self, filepath):
        """
        Parse a FrameView or PresentMon CSV log file.
        Returns a cleaned DataFrame with frame_time_ms and fps columns.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # FrameView CSVs sometimes have comment lines starting with #
        # Try reading with comment handling
        try:
            df = pd.read_csv(filepath, comment='#', on_bad_lines='skip')
        except Exception:
            # Fallback: skip bad lines more aggressively
            df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8-sig')

        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Find the frame time column
        ft_col = None
        for candidate in self.FRAME_TIME_COLS:
            if candidate in df.columns:
                ft_col = candidate
                break

        if ft_col is None:
            available = [c for c in df.columns if 'ms' in c.lower() or 'time' in c.lower() or 'frame' in c.lower()]
            raise ValueError(
                f"No frame time column found in {filepath.name}.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Possible matches: {available}"
            )

        # Convert and clean
        df["frame_time_ms"] = pd.to_numeric(df[ft_col], errors='coerce')
        df = df.dropna(subset=["frame_time_ms"])
        df = df[df["frame_time_ms"] > 0]           # remove zero/negative
        df = df[df["frame_time_ms"] < 1000]         # remove outliers > 1 second

        # Compute FPS
        df["fps"] = 1000.0 / df["frame_time_ms"]

        # Extract render latency if available
        if "MsUntilRenderComplete" in df.columns:
            df["render_latency_ms"] = pd.to_numeric(df["MsUntilRenderComplete"], errors='coerce')

        # Extract display latency if available
        if "MsUntilDisplayed" in df.columns:
            df["display_latency_ms"] = pd.to_numeric(df["MsUntilDisplayed"], errors='coerce')

        # Extract GPU power if available
        for pow_col in ["GPUPower(W)", "GPU0Power(W)", "GPU Power (W)"]:
            if pow_col in df.columns:
                df["gpu_power_w"] = pd.to_numeric(df[pow_col], errors='coerce')
                break

        # Add frame index
        df["frame_num"] = range(len(df))

        # Detect stutters: frame time > 2× median
        median_ft = df["frame_time_ms"].median()
        df["is_stutter"] = df["frame_time_ms"] > (median_ft * 2)

        return df

    def analyze(self, df, label="Unknown"):
        """
        Generate comprehensive statistical analysis from parsed frame data.
        Returns a dict with all performance metrics.
        """
        ft = df["frame_time_ms"]
        fps = df["fps"]

        stats = {
            "label": label,
            "frame_count": len(df),
            "duration_sec": round(ft.sum() / 1000.0, 2),

            # FPS metrics
            "avg_fps": round(fps.mean(), 2),
            "median_fps": round(fps.median(), 2),
            "min_fps": round(fps.min(), 2),
            "max_fps": round(fps.max(), 2),
            "p1_fps": round(fps.quantile(0.01), 2),    # 1% low
            "p5_fps": round(fps.quantile(0.05), 2),    # 5% low
            "p95_fps": round(fps.quantile(0.95), 2),
            "p99_fps": round(fps.quantile(0.99), 2),

            # Frame time metrics
            "avg_frame_time_ms": round(ft.mean(), 3),
            "median_frame_time_ms": round(ft.median(), 3),
            "min_frame_time_ms": round(ft.min(), 3),
            "max_frame_time_ms": round(ft.max(), 3),
            "p99_frame_time_ms": round(ft.quantile(0.99), 3),
            "stdev_frame_time_ms": round(ft.std(), 3),

            # Stutter analysis
            "stutter_count": int(df["is_stutter"].sum()),
            "stutter_pct": round(df["is_stutter"].mean() * 100, 2),
            "worst_frame_time_ms": round(ft.max(), 3),

            # Smoothness: P99/Median ratio (closer to 1.0 = smoother)
            "smoothness_ratio": round(ft.quantile(0.99) / ft.median(), 3) if ft.median() > 0 else 0,

            # Frame time variance coefficient (lower = more consistent)
            "variance_coefficient": round(ft.std() / ft.mean(), 4) if ft.mean() > 0 else 0,
        }

        # Latency stats if available
        if "render_latency_ms" in df.columns:
            rl = df["render_latency_ms"].dropna()
            if len(rl) > 0:
                stats["avg_render_latency_ms"] = round(rl.mean(), 3)
                stats["p99_render_latency_ms"] = round(rl.quantile(0.99), 3)

        # Power stats if available
        if "gpu_power_w" in df.columns:
            gp = df["gpu_power_w"].dropna()
            if len(gp) > 0:
                stats["avg_gpu_power_w"] = round(gp.mean(), 1)
                stats["perf_per_watt"] = round(fps.mean() / gp.mean(), 3) if gp.mean() > 0 else 0

        return stats

    def parse_directory(self, directory):
        """Parse all CSV files in a directory. Returns list of (filename, df, stats)."""
        directory = Path(directory)
        results = []

        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            print(f"  No CSV files found in {directory}/")
            return results

        for csv_path in csv_files:
            try:
                df = self.parse(csv_path)
                stats = self.analyze(df, label=csv_path.stem)
                results.append((csv_path.name, df, stats))
                print(f"  ✓ {csv_path.name}: {stats['frame_count']} frames, "
                      f"Avg {stats['avg_fps']} FPS, "
                      f"1% Low {stats['p1_fps']} FPS, "
                      f"Stutters {stats['stutter_pct']}%")
            except Exception as e:
                print(f"  ✗ {csv_path.name}: {e}")

        return results


# =============================================================================
# Standalone usage
# =============================================================================
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python frameview_parser.py <csv_file_or_directory>")
        print("Example: python frameview_parser.py ../frameview_logs/")
        sys.exit(1)

    target = Path(sys.argv[1])
    parser = FrameViewParser()

    if target.is_dir():
        results = parser.parse_directory(target)
        for name, df, stats in results:
            print(f"\n{'='*50}")
            print(json.dumps(stats, indent=2))
    elif target.is_file():
        df = parser.parse(target)
        stats = parser.analyze(df, label=target.stem)
        print(json.dumps(stats, indent=2))
    else:
        print(f"ERROR: {target} not found")