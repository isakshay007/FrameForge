"""
FrameForge — ai_advisor.py
============================
AI-powered GPU performance advisor using NVIDIA NIM (Llama 3.1) and LangChain.
Ingests benchmark data, identifies bottlenecks, and generates optimization
recommendations — the same workflow NVIDIA's Agentic AI team builds.

Setup:
    1. Get free API key from https://build.nvidia.com
    2. Set environment variable: export NVIDIA_API_KEY=nvapi-XXXXX
    3. pip install langchain-nvidia-ai-endpoints langchain-core langchain

Usage:
    from ai_advisor import PerformanceAdvisor
    advisor = PerformanceAdvisor()
    report = advisor.analyze_cuda_results("results/cuda_results.json")
    report = advisor.analyze_frameview_results(stats_list)
"""

import os
import json
from pathlib import Path


def get_nim_api_key():
    """Get NVIDIA NIM API key from environment."""
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        key = os.environ.get("NVIDIA_NIM_API_KEY", "")
    return key


def check_dependencies():
    """Check if LangChain + NVIDIA packages are installed."""
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return True
    except ImportError:
        return False


class PerformanceAdvisor:
    """
    AI agent that analyzes GPU benchmark data and generates
    optimization recommendations using NVIDIA NIM + LangChain.
    """

    MODEL = "meta/llama-3.1-8b-instruct"

    def __init__(self, api_key=None):
        self.api_key = api_key or get_nim_api_key()
        self.llm = None
        self.available = False

        if not self.api_key:
            print("  [AI Advisor] No NVIDIA API key found.")
            print("  Set: export NVIDIA_API_KEY=nvapi-XXXXX")
            print("  Get key: https://build.nvidia.com")
            return

        if not check_dependencies():
            print("  [AI Advisor] Missing packages. Run:")
            print("  pip install langchain-nvidia-ai-endpoints langchain-core")
            return

        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            self.llm = ChatNVIDIA(
                model=self.MODEL,
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=1024,
            )
            self.available = True
            print(f"  [AI Advisor] Connected to NVIDIA NIM ({self.MODEL})")
        except Exception as e:
            print(f"  [AI Advisor] Failed to connect: {e}")

    def _query(self, prompt):
        """Send a prompt to NVIDIA NIM and return the response."""
        if not self.available:
            return self._fallback_analysis(prompt)

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=(
                    "You are an expert NVIDIA GPU performance engineer. "
                    "Analyze the benchmark data provided and give specific, "
                    "actionable optimization recommendations. Be concise and technical. "
                    "Focus on: bottlenecks, memory bandwidth issues, compute vs memory bound, "
                    "resolution scaling efficiency, and stutter causes."
                )),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            print(f"  [AI Advisor] API call failed: {e}")
            return self._fallback_analysis(prompt)

    def _fallback_analysis(self, prompt):
        """Rule-based fallback when NIM API is unavailable."""
        return (
            "⚠️ AI Advisor unavailable (no API key or connection error).\n"
            "Rule-based analysis: Review the benchmark data manually.\n"
            "Key things to check:\n"
            "- Frame times increasing >15% between resolutions = GPU bottleneck\n"
            "- Stutter rate >5% = inconsistent frame pacing\n"
            "- Smoothness ratio >2.0 = significant frame time spikes\n"
            "- P1 FPS much lower than Avg FPS = periodic hitching\n"
            "Set NVIDIA_API_KEY for AI-powered analysis."
        )

    def analyze_cuda_results(self, cuda_json_path):
        """Analyze C++ CUDA benchmark results with AI."""
        path = Path(cuda_json_path)
        if not path.exists():
            return {"error": f"File not found: {path}"}

        with open(path) as f:
            data = json.load(f)

        # Build analysis prompt
        gpu = data.get("gpu", "Unknown")
        vram = data.get("vram_mb", "?")
        results = data.get("results", [])

        results_text = ""
        for r in results:
            results_text += (
                f"  {r['kernel']} @ {r['resolution']}: "
                f"avg={r['avg_ms']:.2f}ms, min={r['min_ms']:.2f}ms, "
                f"max={r['max_ms']:.2f}ms, throughput={r['throughput']:.1f} {r['unit']}\n"
            )

        prompt = f"""Analyze these CUDA shader benchmark results from GPU: {gpu} ({vram}MB VRAM)

Benchmarks simulate gaming GPU pipeline stages:
- vertex_shader: MVP matrix transform per vertex
- pixel_shader: Blinn-Phong per-pixel lighting
- postfx_blur: 5x5 Gaussian post-processing blur
- texture_sample: Bilinear texture interpolation

Results:
{results_text}

Provide:
1. Which kernel is the bottleneck at each resolution?
2. Is the GPU compute-bound or memory-bound for each kernel?
3. How does performance scale from 1080p to 4K? Is the scaling expected?
4. What specific optimizations would you recommend for the worst-performing kernel?
5. Overall assessment: is this GPU suitable for 4K gaming based on these results?"""

        print("  [AI Advisor] Analyzing CUDA benchmark results...")
        analysis = self._query(prompt)

        report = {
            "type": "cuda_analysis",
            "gpu": gpu,
            "model": self.MODEL if self.available else "fallback",
            "analysis": analysis,
        }

        return report

    def analyze_frameview_results(self, stats_list):
        """Analyze FrameView benchmark statistics with AI."""
        if not stats_list:
            return {"error": "No FrameView stats provided"}

        stats_text = ""
        for s in stats_list:
            stats_text += (
                f"  {s['label']}:\n"
                f"    Avg FPS: {s['avg_fps']}, 1% Low: {s['p1_fps']}, "
                f"P99 Frame Time: {s['p99_frame_time_ms']}ms\n"
                f"    Stutter: {s['stutter_pct']}%, "
                f"Smoothness: {s['smoothness_ratio']}\n"
            )

        prompt = f"""Analyze these gaming performance benchmarks (FrameView data):

{stats_text}

Provide:
1. Which resolution/setting has the worst performance and why?
2. Are there stutter issues? What's likely causing them?
3. Is the frame pacing consistent (check smoothness ratio)?
4. What in-game settings should the user lower first to improve FPS?
5. Would enabling DLSS or FSR help? At which resolutions?
6. Overall: rate the gaming experience at each setting (Excellent/Good/Playable/Poor)."""

        print("  [AI Advisor] Analyzing FrameView performance data...")
        analysis = self._query(prompt)

        report = {
            "type": "frameview_analysis",
            "model": self.MODEL if self.available else "fallback",
            "num_benchmarks": len(stats_list),
            "analysis": analysis,
        }

        return report

    def analyze_regressions(self, regressions):
        """Analyze detected performance regressions with AI."""
        if not regressions:
            return {"analysis": "No regressions detected. Performance is consistent."}

        reg_text = ""
        for r in regressions:
            if "kernel" in r:
                reg_text += (
                    f"  [{r['severity']}] {r['kernel']}: "
                    f"{r['from_res']}→{r['to_res']} "
                    f"+{r['change_pct']}% frame time increase\n"
                )
            else:
                reg_text += (
                    f"  [{r.get('severity', '?')}] {r.get('metric', '?')}: "
                    f"{r.get('change_pct', 0):+.1f}% change\n"
                )

        prompt = f"""These GPU performance regressions were detected:

{reg_text}

For each regression:
1. What is the likely root cause?
2. Is this regression expected (e.g., 4K naturally slower) or unexpected?
3. What hardware or software fix would address it?
4. Priority ranking: which regression should be investigated first?"""

        print("  [AI Advisor] Analyzing performance regressions...")
        analysis = self._query(prompt)

        return {
            "type": "regression_analysis",
            "num_regressions": len(regressions),
            "analysis": analysis,
        }

    def generate_full_report(self, cuda_json=None, fv_stats=None, regressions=None):
        """Generate a comprehensive AI performance report."""
        print("\n[AI ADVISOR] Generating full performance report...")

        sections = []

        if cuda_json and Path(cuda_json).exists():
            cuda_report = self.analyze_cuda_results(cuda_json)
            sections.append(("CUDA Shader Benchmark Analysis", cuda_report["analysis"]))

        if fv_stats:
            fv_report = self.analyze_frameview_results(fv_stats)
            sections.append(("FrameView Performance Analysis", fv_report["analysis"]))

        if regressions:
            reg_report = self.analyze_regressions(regressions)
            sections.append(("Regression Root Cause Analysis", reg_report["analysis"]))

        # Compile full report
        full_report = "=" * 60 + "\n"
        full_report += "  FrameForge AI Performance Report\n"
        full_report += f"  Powered by NVIDIA NIM ({self.MODEL})\n" if self.available else "  (Fallback mode)\n"
        full_report += "=" * 60 + "\n\n"

        for title, content in sections:
            full_report += f"## {title}\n\n{content}\n\n{'—' * 40}\n\n"

        return full_report


# =============================================================================
# Standalone usage
# =============================================================================
if __name__ == "__main__":
    import sys

    advisor = PerformanceAdvisor()

    if len(sys.argv) > 1:
        cuda_path = sys.argv[1]
        report = advisor.analyze_cuda_results(cuda_path)
        print(report["analysis"])
    else:
        print("Usage: python ai_advisor.py <cuda_results.json>")
        print("  Set NVIDIA_API_KEY environment variable first.")
        print("  Get key: https://build.nvidia.com")