"""
FrameForge — regression_detector.py
====================================
Compares benchmark runs and detects performance regressions.
Flags frame time increases: >10% = MEDIUM, >20% = HIGH.

Usage:
    from regression_detector import RegressionDetector
    detector = RegressionDetector()
    report = detector.compare(stats_a, stats_b)
"""

import json
from pathlib import Path


class RegressionDetector:
    """Compares benchmark results and detects performance regressions."""

    MEDIUM_THRESHOLD = 10.0   # % frame time increase
    HIGH_THRESHOLD = 20.0     # % frame time increase

    def compare(self, baseline_stats, test_stats):
        """
        Compare two benchmark runs.
        baseline_stats, test_stats: dicts from FrameViewParser.analyze()
        Returns a regression report dict.
        """
        report = {
            "baseline": baseline_stats.get("label", "Baseline"),
            "test": test_stats.get("label", "Test"),
            "metrics": {},
            "regressions": [],
            "improvements": [],
            "summary": "",
        }

        # Metrics to compare
        comparisons = [
            ("avg_fps", "higher_better"),
            ("p1_fps", "higher_better"),
            ("p5_fps", "higher_better"),
            ("avg_frame_time_ms", "lower_better"),
            ("p99_frame_time_ms", "lower_better"),
            ("stutter_pct", "lower_better"),
            ("smoothness_ratio", "lower_better"),
        ]

        for metric, direction in comparisons:
            base_val = baseline_stats.get(metric)
            test_val = test_stats.get(metric)

            if base_val is None or test_val is None or base_val == 0:
                continue

            pct_change = ((test_val - base_val) / abs(base_val)) * 100
            is_regression = (
                (direction == "higher_better" and pct_change < -self.MEDIUM_THRESHOLD) or
                (direction == "lower_better" and pct_change > self.MEDIUM_THRESHOLD)
            )
            is_improvement = (
                (direction == "higher_better" and pct_change > self.MEDIUM_THRESHOLD) or
                (direction == "lower_better" and pct_change < -self.MEDIUM_THRESHOLD)
            )

            severity = "OK"
            if is_regression:
                abs_change = abs(pct_change)
                severity = "HIGH" if abs_change > self.HIGH_THRESHOLD else "MEDIUM"

            entry = {
                "metric": metric,
                "baseline": base_val,
                "test": test_val,
                "change_pct": round(pct_change, 2),
                "direction": direction,
                "severity": severity,
            }
            report["metrics"][metric] = entry

            if is_regression:
                report["regressions"].append(entry)
            elif is_improvement:
                report["improvements"].append(entry)

        # Summary
        n_reg = len(report["regressions"])
        n_imp = len(report["improvements"])
        high_count = sum(1 for r in report["regressions"] if r["severity"] == "HIGH")

        if n_reg == 0:
            report["summary"] = "No regressions detected."
        elif high_count > 0:
            report["summary"] = f"🔴 {n_reg} regressions ({high_count} HIGH severity)"
        else:
            report["summary"] = f"🟡 {n_reg} regressions (MEDIUM severity)"

        if n_imp > 0:
            report["summary"] += f" | {n_imp} improvements"

        return report

    def compare_resolutions(self, stats_list):
        """
        Compare across multiple resolutions (e.g., 1080p → 1440p → 4K).
        stats_list: list of dicts from FrameViewParser.analyze()
        Returns list of regression reports between consecutive resolutions.
        """
        reports = []
        for i in range(len(stats_list) - 1):
            report = self.compare(stats_list[i], stats_list[i + 1])
            reports.append(report)
        return reports

    def compare_cuda_results(self, cuda_json_path):
        """
        Analyze CUDA benchmark results for cross-resolution regressions.
        cuda_json_path: path to JSON output from cuda_bench
        """
        path = Path(cuda_json_path)
        if not path.exists():
            print(f"  CUDA results not found: {path}")
            return []

        with open(path) as f:
            data = json.load(f)

        results = data.get("results", [])
        if not results:
            return []

        # Group by kernel
        kernels = {}
        for r in results:
            k = r["kernel"]
            if k not in kernels:
                kernels[k] = []
            kernels[k].append(r)

        reports = []
        for kernel_name, entries in kernels.items():
            # Sort by resolution order
            res_order = {"1080p": 0, "1440p": 1, "4K": 2}
            entries.sort(key=lambda x: res_order.get(x["resolution"], 99))

            for i in range(len(entries) - 1):
                base = entries[i]
                test = entries[i + 1]

                ft_change = ((test["avg_ms"] - base["avg_ms"]) / base["avg_ms"]) * 100
                severity = "OK"
                if ft_change > self.HIGH_THRESHOLD:
                    severity = "HIGH"
                elif ft_change > self.MEDIUM_THRESHOLD:
                    severity = "MEDIUM"

                if severity != "OK":
                    reports.append({
                        "kernel": kernel_name,
                        "from_res": base["resolution"],
                        "to_res": test["resolution"],
                        "base_ms": base["avg_ms"],
                        "test_ms": test["avg_ms"],
                        "change_pct": round(ft_change, 1),
                        "severity": severity,
                    })

        return reports

    @staticmethod
    def print_report(report):
        """Pretty-print a regression report."""
        print(f"\n{'='*60}")
        print(f"  Regression Report: {report['baseline']} → {report['test']}")
        print(f"{'='*60}")
        print(f"  {report['summary']}")

        if report["regressions"]:
            print(f"\n    Regressions:")
            for r in report["regressions"]:
                print(f"    [{r['severity']}] {r['metric']}: "
                      f"{r['baseline']} → {r['test']} ({r['change_pct']:+.1f}%)")

        if report["improvements"]:
            print(f"\n   Improvements:")
            for r in report["improvements"]:
                print(f"    {r['metric']}: {r['baseline']} → {r['test']} ({r['change_pct']:+.1f}%)")

        print()


# =============================================================================
# Standalone usage
# =============================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python regression_detector.py <cuda_results.json>")
        print("       python regression_detector.py --compare <stats_a.json> <stats_b.json>")
        sys.exit(1)

    detector = RegressionDetector()

    if sys.argv[1] == "--compare" and len(sys.argv) >= 4:
        with open(sys.argv[2]) as f:
            a = json.load(f)
        with open(sys.argv[3]) as f:
            b = json.load(f)
        report = detector.compare(a, b)
        detector.print_report(report)
    else:
        reports = detector.compare_cuda_results(sys.argv[1])
        if reports:
            print(f"\n⚠️  {len(reports)} CUDA regressions detected:")
            for r in reports:
                print(f"  [{r['severity']}] {r['kernel']}: "
                      f"{r['from_res']}→{r['to_res']} "
                      f"{r['change_pct']:+.1f}% ({r['base_ms']:.2f}→{r['test_ms']:.2f}ms)")
        else:
            print(" No CUDA regressions detected.")