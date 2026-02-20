"""
FrameForge — dashboard.py
===========================
Generates publication-quality performance dashboards from benchmark data.
Creates 6 chart types: FPS comparison, frame time distribution, stutter
timeline, CUDA throughput, quality scaling, and regression report.

Usage:
    from dashboard import DashboardGenerator
    gen = DashboardGenerator(output_dir="results")
    gen.plot_fps_comparison(stats_list)
    gen.plot_frame_time_timeline(df)
    gen.plot_cuda_results(cuda_json)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class DashboardGenerator:
    """Generates Matplotlib performance dashboards."""

    COLORS = {
        "1080p": "#00d26a", "1440p": "#f5a623", "4K": "#e74c3c",
        "vertex_shader": "#4ecdc4", "pixel_shader": "#f5a623",
        "postfx_blur": "#e74c3c", "texture_sample": "#9b59b6",
    }
    BG = "#1a1a2e"
    FG = "#e0e0e0"

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        plt.rcParams.update({
            "figure.facecolor": self.BG, "axes.facecolor": "#16213e",
            "axes.edgecolor": "#2a2a4a", "axes.labelcolor": self.FG,
            "text.color": self.FG, "xtick.color": self.FG,
            "ytick.color": self.FG, "grid.color": "#2a2a4a",
            "font.size": 10, "font.family": "sans-serif",
        })

    def _save(self, fig, name):
        path = self.output_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self.BG)
        plt.close(fig)
        print(f"  ✓ Saved: {path}")

    # ----- Chart 1: FPS Comparison Bar Chart -----
    def plot_fps_comparison(self, stats_list):
        """Bar chart comparing FPS across multiple benchmark runs."""
        if not stats_list:
            return

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle("FPS Comparison Across Runs", fontsize=16, fontweight="bold")

        labels = [s["label"] for s in stats_list]
        avg_fps = [s["avg_fps"] for s in stats_list]
        p1_fps = [s["p1_fps"] for s in stats_list]

        x = np.arange(len(labels))
        w = 0.35

        bars1 = ax.bar(x - w/2, avg_fps, w, label="Avg FPS", color="#00d26a", edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + w/2, p1_fps, w, label="1% Low FPS", color="#e74c3c", edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars1, avg_fps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
        for bar, val in zip(bars2, p1_fps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("FPS")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        self._save(fig, "fps_comparison")

    # ----- Chart 2: Frame Time Distribution -----
    def plot_frame_time_distribution(self, dataframes, labels):
        """Histogram of frame time distributions for multiple runs."""
        if not dataframes:
            return

        fig, ax = plt.subplots(figsize=(14, 7))
        fig.suptitle("Frame Time Distribution", fontsize=16, fontweight="bold")

        colors = list(self.COLORS.values())
        for i, (df, label) in enumerate(zip(dataframes, labels)):
            color = colors[i % len(colors)]
            ax.hist(df["frame_time_ms"], bins=60, alpha=0.5, label=label,
                    color=color, edgecolor="white", linewidth=0.3)

        ax.set_xlabel("Frame Time (ms)")
        ax.set_ylabel("Frame Count")
        ax.legend()
        ax.grid(alpha=0.3)

        self._save(fig, "frame_time_distribution")

    # ----- Chart 3: Frame Time Timeline (Stutter Detection) -----
    def plot_frame_time_timeline(self, df, label="Benchmark"):
        """Line chart showing per-frame times with stutter markers."""
        if df is None or df.empty:
            return

        fig, ax = plt.subplots(figsize=(16, 7))
        fig.suptitle(f"Frame Time Timeline — {label}", fontsize=16, fontweight="bold")

        ft = df["frame_time_ms"].values
        median_ft = np.median(ft)

        ax.plot(ft, color="#00d26a", alpha=0.8, linewidth=0.6)
        ax.axhline(median_ft, color="white", linestyle="--", alpha=0.4,
                    label=f"Median: {median_ft:.1f}ms")
        ax.axhline(median_ft * 2, color="#e74c3c", linestyle="--", alpha=0.4,
                    label=f"Stutter threshold: {median_ft*2:.1f}ms")

        # Mark stutters
        stutter_mask = ft > median_ft * 2
        if stutter_mask.any():
            stutter_idx = np.where(stutter_mask)[0]
            ax.scatter(stutter_idx, ft[stutter_mask], color="#e74c3c",
                       s=20, zorder=5, marker="x", label=f"Stutters ({stutter_mask.sum()})")

        ax.set_xlabel("Frame #")
        ax.set_ylabel("Frame Time (ms)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        self._save(fig, "stutter_timeline")

    # ----- Chart 4: CUDA Kernel Throughput -----
    def plot_cuda_results(self, cuda_json_path):
        """Bar chart of CUDA shader pipeline throughput by resolution."""
        path = Path(cuda_json_path)
        if not path.exists():
            print(f"  [SKIP] CUDA results not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(f"CUDA Shader Pipeline — {data.get('gpu', 'Unknown GPU')}",
                     fontsize=16, fontweight="bold")

        # Chart 4a: Execution time by kernel × resolution
        ax = axes[0]
        kernels = sorted(set(r["kernel"] for r in results))
        resolutions = ["1080p", "1440p", "4K"]
        x = np.arange(len(kernels))
        w = 0.25

        for i, res in enumerate(resolutions):
            times = []
            for k in kernels:
                match = [r for r in results if r["kernel"] == k and r["resolution"] == res]
                times.append(match[0]["avg_ms"] if match else 0)
            ax.bar(x + i*w, times, w, label=res, color=self.COLORS.get(res, "#999"),
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x + w)
        ax.set_xticklabels([k.replace("_", "\n") for k in kernels], fontsize=9)
        ax.set_ylabel("Execution Time (ms)")
        ax.set_title("Kernel Execution Time", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Chart 4b: Throughput by kernel × resolution
        ax = axes[1]
        for i, res in enumerate(resolutions):
            tps = []
            for k in kernels:
                match = [r for r in results if r["kernel"] == k and r["resolution"] == res]
                tps.append(match[0]["throughput"] if match else 0)
            ax.bar(x + i*w, tps, w, label=res, color=self.COLORS.get(res, "#999"),
                   edgecolor="white", linewidth=0.5)

        ax.set_xticks(x + w)
        ax.set_xticklabels([k.replace("_", "\n") for k in kernels], fontsize=9)
        ax.set_ylabel("Throughput")
        ax.set_title("Kernel Throughput", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        self._save(fig, "cuda_throughput")

    # ----- Chart 5: Regression Severity Report -----
    def plot_regression_report(self, regressions, title="Performance Regressions"):
        """Horizontal bar chart of detected regressions."""
        fig, ax = plt.subplots(figsize=(12, max(4, len(regressions) * 0.8 + 2)))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        if not regressions:
            ax.text(0.5, 0.5, " No Regressions Detected",
                    ha="center", va="center", fontsize=20, color="#00d26a",
                    transform=ax.transAxes)
            ax.axis("off")
        else:
            labels = []
            values = []
            colors = []
            for r in regressions:
                if "kernel" in r:  # CUDA regression
                    labels.append(f"{r['kernel']}\n{r['from_res']}→{r['to_res']}")
                    values.append(r["change_pct"])
                else:  # FrameView regression
                    labels.append(f"{r['metric']}")
                    values.append(abs(r["change_pct"]))

                severity = r.get("severity", "MEDIUM")
                colors.append("#e74c3c" if severity == "HIGH" else "#f5a623")

            bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
            ax.axvline(10, color="#f5a623", linestyle="--", alpha=0.5, label="MEDIUM (10%)")
            ax.axvline(20, color="#e74c3c", linestyle="--", alpha=0.5, label="HIGH (20%)")
            ax.set_xlabel("Frame Time Increase (%)")
            ax.legend(loc="lower right")
            ax.grid(axis="x", alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=10)

        self._save(fig, "regression_report")

    # ----- Chart 6: Summary Stats Table -----
    def plot_summary_table(self, stats_list):
        """Visual summary table of all benchmark results."""
        if not stats_list:
            return

        fig, ax = plt.subplots(figsize=(14, max(3, len(stats_list) * 0.6 + 2)))
        fig.suptitle("Benchmark Summary", fontsize=16, fontweight="bold")
        ax.axis("off")

        cols = ["Run", "Avg FPS", "1% Low", "P99 FT(ms)", "Stutters", "Smoothness"]
        rows = []
        for s in stats_list:
            rows.append([
                s["label"],
                f"{s['avg_fps']:.1f}",
                f"{s['p1_fps']:.1f}",
                f"{s['p99_frame_time_ms']:.1f}",
                f"{s['stutter_pct']:.1f}%",
                f"{s['smoothness_ratio']:.2f}",
            ])

        table = ax.table(cellText=rows, colLabels=cols, loc="center",
                         cellLoc="center", colColours=["#2a2a4a"]*len(cols))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        # Style cells
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#3a3a5a")
            if row == 0:
                cell.set_facecolor("#2a2a4a")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#16213e")
                cell.set_text_props(color="#e0e0e0")

        self._save(fig, "summary_table")


# =============================================================================
# Standalone usage
# =============================================================================
if __name__ == "__main__":
    print("Dashboard generator — use via main.py or import directly.")
    print("Example: python main.py --dashboard")