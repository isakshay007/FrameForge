"""
FrameForge — ai_advisor.py (Updated)
======================================
AI-powered GPU performance advisor supporting multiple backends:
  1. DeepSeek R1 (reasoning model) — recommended, free 5M tokens
  2. NVIDIA NIM (Llama 3.1) via LangChain — free 1000 credits
  3. Rule-based fallback — no API key needed

Setup:
    DeepSeek R1 (recommended):
        1. Sign up at https://platform.deepseek.com
        2. Get API key from dashboard → API Keys
        3. pip install openai
        4. Set: export DEEPSEEK_API_KEY=sk-XXXXX

    NVIDIA NIM (alternative):
        1. Sign up at https://build.nvidia.com
        2. Get API key from settings → API Keys
        3. pip install langchain-nvidia-ai-endpoints langchain-core
        4. Set: export NVIDIA_API_KEY=nvapi-XXXXX

Usage:
    from ai_advisor import PerformanceAdvisor

    # Auto-detects available backend (DeepSeek → NIM → Fallback)
    advisor = PerformanceAdvisor()

    # Or specify backend explicitly
    advisor = PerformanceAdvisor(backend="deepseek", api_key="sk-xxx")
    advisor = PerformanceAdvisor(backend="nvidia", api_key="nvapi-xxx")

    # Analyze real FrameView data
    report = advisor.analyze_frameview_results(stats_list)

    # Full report (FrameView + CUDA + Regressions)
    report = advisor.generate_full_report(
        cuda_json="results/cuda_results.json",
        fv_stats=stats_list,
        regressions=all_regs
    )
"""

import os
import json
import time
from pathlib import Path


# API Key Detection — reads keys from environment variables

def get_deepseek_key():
    """Read DeepSeek API key from DEEPSEEK_API_KEY environment variable."""
    return os.environ.get("DEEPSEEK_API_KEY", "")


def get_nim_key():
    """Read NVIDIA NIM API key from NVIDIA_API_KEY or NVIDIA_NIM_API_KEY."""
    return os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NVIDIA_NIM_API_KEY", "")


def check_openai_sdk():
    """Check if the OpenAI Python SDK is installed (required for DeepSeek)."""
    try:
        from openai import OpenAI
        return True
    except ImportError:
        return False


def check_langchain_nvidia():
    """Check if LangChain NVIDIA endpoints package is installed (required for NIM)."""
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return True
    except ImportError:
        return False


# PerformanceAdvisor — Multi-Backend AI Analysis Engine
#
# Supports three backends in priority order:
#   1. DeepSeek R1   — deep reasoning, best analysis quality
#   2. NVIDIA NIM    — fast inference via LangChain
#   3. Fallback      — rule-based checks, no API needed

class PerformanceAdvisor:
    """
    AI agent that analyzes GPU benchmark data and generates
    optimization recommendations using DeepSeek R1, NVIDIA NIM,
    or a rule-based fallback when no API key is available.
    """

    # Model identifiers for each backend
    DEEPSEEK_MODEL = "deepseek-reasoner"       # DeepSeek R1 — reasoning model
    DEEPSEEK_FAST_MODEL = "deepseek-chat"      # DeepSeek V3 — faster, cheaper
    NIM_MODEL = "meta/llama-3.1-8b-instruct"   # NVIDIA NIM — Llama 3.1 8B

    def __init__(self, backend=None, api_key=None, use_reasoning=True):
        """
        Initialize the advisor with the best available AI backend.

        Args:
            backend:       "deepseek", "nvidia", or None (auto-detect best available)
            api_key:       API key string, or None to read from environment
            use_reasoning: True = DeepSeek R1 (slower, deeper), False = V3 Chat (faster)
        """
        self.backend = backend
        self.api_key = api_key
        self.client = None
        self.available = False
        self.model_name = ""
        self.use_reasoning = use_reasoning
        self.last_reasoning = None  # stores R1's internal reasoning chain

        # Auto-detect backend if not specified
        if self.backend is None:
            self.backend = self._auto_detect_backend()

        # Initialize the chosen backend
        if self.backend == "deepseek":
            self._init_deepseek()
        elif self.backend == "nvidia":
            self._init_nvidia()
        else:
            print("  [AI Advisor] No API keys found. Using rule-based fallback.")
            print("  For AI analysis, set one of:")
            print("    export DEEPSEEK_API_KEY=sk-XXXXX  (recommended)")
            print("    export NVIDIA_API_KEY=nvapi-XXXXX")
            self.backend = "fallback"

    def _auto_detect_backend(self):
        """Auto-detect the best available backend based on API key format."""
        key = self.api_key

        # Check if a key was passed directly
        if key and key.startswith("sk-"):
            return "deepseek"
        if key and key.startswith("nvapi-"):
            return "nvidia"

        # Check environment variables
        if get_deepseek_key():
            self.api_key = get_deepseek_key()
            return "deepseek"
        if get_nim_key():
            self.api_key = get_nim_key()
            return "nvidia"

        return "fallback"

    def _init_deepseek(self):
        """Initialize DeepSeek R1/V3 backend using OpenAI-compatible SDK."""
        if not self.api_key:
            self.api_key = get_deepseek_key()

        if not self.api_key:
            print("  [AI Advisor] No DeepSeek API key found.")
            print("  Get one free at: https://platform.deepseek.com")
            self.backend = "fallback"
            return

        if not check_openai_sdk():
            print("  [AI Advisor] OpenAI SDK not found. Run: pip install openai")
            self.backend = "fallback"
            return

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
            self.model_name = self.DEEPSEEK_MODEL if self.use_reasoning else self.DEEPSEEK_FAST_MODEL
            self.available = True
            mode = "R1 Reasoning" if self.use_reasoning else "V3 Chat"
            print(f"  [AI Advisor] Connected to DeepSeek {mode} ({self.model_name})")
        except Exception as e:
            print(f"  [AI Advisor] DeepSeek connection failed: {e}")
            self.backend = "fallback"

    def _init_nvidia(self):
        """Initialize NVIDIA NIM backend using LangChain."""
        if not self.api_key:
            self.api_key = get_nim_key()

        if not self.api_key:
            print("  [AI Advisor] No NVIDIA API key found.")
            print("  Get one free at: https://build.nvidia.com")
            self.backend = "fallback"
            return

        if not check_langchain_nvidia():
            print("  [AI Advisor] LangChain NVIDIA package not found. Run:")
            print("  pip install langchain-nvidia-ai-endpoints langchain-core")
            self.backend = "fallback"
            return

        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            self.client = ChatNVIDIA(
                model=self.NIM_MODEL,
                api_key=self.api_key,
                temperature=0.3,
                max_tokens=2048,
            )
            self.model_name = self.NIM_MODEL
            self.available = True
            print(f"  [AI Advisor] Connected to NVIDIA NIM ({self.model_name})")
        except Exception as e:
            print(f"  [AI Advisor] NVIDIA NIM connection failed: {e}")
            self.backend = "fallback"


    # Core Query Methods — route prompts to the active backend

    def _query(self, prompt, system_prompt=None):
        """Send a prompt to the active backend and return the AI response."""
        if not self.available:
            return self._fallback_analysis(prompt)

        default_system = (
            "You are an expert NVIDIA GPU performance engineer. "
            "Analyze the benchmark data provided and give specific, "
            "actionable optimization recommendations. Be concise and technical. "
            "Focus on: bottlenecks, memory bandwidth, compute vs memory bound, "
            "resolution scaling efficiency, frame pacing, and stutter causes."
        )
        system = system_prompt or default_system

        try:
            if self.backend == "deepseek":
                return self._query_deepseek(prompt, system)
            elif self.backend == "nvidia":
                return self._query_nvidia(prompt, system)
            else:
                return self._fallback_analysis(prompt)
        except Exception as e:
            print(f"  [AI Advisor] API call failed: {e}")
            return self._fallback_analysis(prompt)

    def _query_deepseek(self, prompt, system):
        """Query DeepSeek API. R1 doesn't support system messages so we prepend to user."""
        if self.use_reasoning:
            # R1 (reasoner) — no system role allowed, merge into user message
            messages = [{"role": "user", "content": f"{system}\n\n{prompt}"}]
        else:
            # V3 (chat) — standard system + user format
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]

        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=4000
        )
        elapsed = time.time() - t0

        # Capture R1's internal reasoning chain if available
        self.last_reasoning = None
        if hasattr(response.choices[0].message, 'reasoning_content'):
            self.last_reasoning = response.choices[0].message.reasoning_content

        content = response.choices[0].message.content
        print(f"  [AI Advisor] Response received ({elapsed:.1f}s)")
        return content

    def _query_nvidia(self, prompt, system):
        """Query NVIDIA NIM via LangChain message format."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]

        t0 = time.time()
        response = self.client.invoke(messages)
        elapsed = time.time() - t0

        print(f"  [AI Advisor] Response received ({elapsed:.1f}s)")
        return response.content

    def _fallback_analysis(self, prompt):
        """Rule-based fallback when no AI backend is available."""
        return (
            "AI Advisor unavailable (no API key or connection error).\n\n"
            "Rule-based analysis — key things to check:\n"
            "- GPU Utilization >95% = GPU-bound (expected at high settings)\n"
            "- CPU Utilization <60% with GPU >95% = GPU is the bottleneck, not CPU\n"
            "- Frame times increasing >15% between resolutions = normal GPU scaling\n"
            "- Stutter rate >5% = inconsistent frame pacing, investigate drivers\n"
            "- Smoothness ratio >2.0 = significant frame time spikes\n"
            "- P1 FPS much lower than Avg FPS = periodic hitching\n"
            "- PC Latency >50ms with VSync ON = consider NVIDIA Reflex or VSync OFF\n"
            "- CPU over TDP = may cause thermal throttling under sustained load\n\n"
            "For AI-powered analysis, set one of:\n"
            "  export DEEPSEEK_API_KEY=sk-XXXXX  (recommended, free)\n"
            "  export NVIDIA_API_KEY=nvapi-XXXXX  (free 1000 credits)\n"
        )


    # Analysis Methods — each builds a targeted prompt and queries the AI

    def analyze_frameview_results(self, stats_list):
        """Analyze FrameView parsed stats (basic mode — uses stats dicts only)."""
        if not stats_list:
            return {"error": "No FrameView stats provided"}

        # Build stats summary text from each benchmark run
        stats_text = ""
        for s in stats_list:
            stats_text += (
                f"\n  {s.get('label', 'Unknown')}:\n"
                f"    Frames: {s.get('frame_count', '?')} | "
                f"Duration: {s.get('duration_sec', '?')}s\n"
                f"    Avg FPS: {s.get('avg_fps', '?')} | "
                f"Median: {s.get('median_fps', '?')} | "
                f"1% Low: {s.get('p1_fps', '?')} | "
                f"5% Low: {s.get('p5_fps', '?')}\n"
                f"    Frame Time: Avg {s.get('avg_frame_time_ms', '?')}ms | "
                f"P99: {s.get('p99_frame_time_ms', '?')}ms | "
                f"StdDev: {s.get('stdev_frame_time_ms', '?')}ms\n"
                f"    Stutter: {s.get('stutter_pct', '?')}% | "
                f"Smoothness Ratio: {s.get('smoothness_ratio', '?')}\n"
                f"    Variance Coefficient: {s.get('variance_coefficient', '?')}\n"
            )

        prompt = f"""Analyze these gaming performance benchmarks (FrameView data):

{stats_text}

Provide:
1. Which resolution/setting has the worst performance and why?
2. Are there stutter issues? What's likely causing them?
3. Is the frame pacing consistent (check smoothness ratio and variance)?
4. What in-game settings should the user lower first to improve FPS?
5. Would enabling DLSS or FSR help? At which resolutions?
6. Overall: rate the gaming experience at each setting (Excellent/Good/Playable/Poor)."""

        print("  [AI Advisor] Analyzing FrameView performance data...")
        analysis = self._query(prompt)

        return {
            "type": "frameview_analysis",
            "backend": self.backend,
            "model": self.model_name,
            "num_benchmarks": len(stats_list),
            "analysis": analysis,
            "reasoning": self.last_reasoning,
        }

    def analyze_frameview_detailed(self, data_dict):
        """
        Enhanced analysis using raw DataFrames for deeper insights.
        This extracts GPU/CPU/latency/per-core data directly from CSVs
        instead of relying on pre-parsed stats.

        Args:
            data_dict: {"1080p": DataFrame, "1440p": DataFrame}
        """
        import numpy as np

        sections = []
        last_df = None

        for label, df in data_dict.items():
            last_df = df
            ft = df["MsBetweenPresents"].dropna()
            fps = 1000 / ft

            # Extract available hardware metrics
            gpu_util = df.get("GPU0Util(%)", None)
            gpu_temp = df.get("GPU0Temp(C)", None)
            gpu_clk = df.get("GPU0Clk(MHz)", None)
            gpu_pwr = df.get("NV Pwr(W) (API)", None)
            cpu_util = df.get("CPUUtil(%)", None)
            cpu_temp = df.get("CPU Package Temp(C)", None)
            cpu_pwr = df.get("CPU Package Power(W)", None)
            pc_lat = df.get("MsPCLatency", None)
            render_lat = df.get("MsRenderPresentLatency", None)

            # Build per-resolution stats block
            section = f"[{label}] Resolution: {df['Resolution'].iloc[0]}\n"
            section += f"  Frames: {len(ft)} | Duration: {(df['TimeInSeconds'].iloc[-1] - df['TimeInSeconds'].iloc[0]):.1f}s\n"
            section += f"  FPS: Avg {fps.mean():.1f} | Median {fps.median():.1f} | Min {fps.min():.1f}\n"
            section += f"  FPS Lows: 0.1% {np.percentile(fps, 0.1):.1f} | 1% {np.percentile(fps, 1):.1f} | 5% {np.percentile(fps, 5):.1f}\n"
            section += f"  Frame Time: Avg {ft.mean():.2f}ms | 99th {np.percentile(ft, 99):.2f}ms | StdDev {ft.std():.2f}ms\n"

            # Stutter and dropped frame counts
            stutters = (ft > ft.mean() * 2).sum()
            dropped = df["Dropped"].sum() if "Dropped" in df.columns else 0
            section += f"  Stutters: {stutters} mild | {int(dropped)} dropped\n"

            # GPU metrics (clock, temp, utilization, power)
            if gpu_util is not None:
                section += f"  GPU: {gpu_util.dropna().mean():.1f}% util | {gpu_clk.dropna().mean():.0f} MHz | {gpu_temp.dropna().mean():.1f}°C | {gpu_pwr.dropna().mean():.0f}W\n"

            # CPU metrics (utilization, temp, power vs TDP)
            if cpu_util is not None:
                section += f"  CPU: {cpu_util.dropna().mean():.1f}% util | {cpu_temp.dropna().mean():.1f}°C | {cpu_pwr.dropna().mean():.1f}W\n"

            # Latency pipeline (PC latency + render latency)
            if pc_lat is not None:
                section += f"  Latency: PC {pc_lat.dropna().mean():.1f}ms | Render {render_lat.dropna().mean():.1f}ms\n"

            # Per-core CPU utilization (only cores with data)
            core_cols = [c for c in df.columns if "CPUCoreUtil%" in c and df[c].notna().any()]
            if core_cols:
                core_avgs = [f"{df[c].dropna().mean():.0f}%" for c in core_cols]
                section += f"  Per-Core CPU: {', '.join(core_avgs)}\n"

            sections.append(section)

        full_data = "\n".join(sections)

        # Use the last DataFrame for system info
        df = last_df

        prompt = f"""You are an expert NVIDIA GPU performance engineer.
Analyze this REAL FrameView benchmark data from an actual gaming PC.

SYSTEM: {df['GPU'].iloc[0]} + {df['CPU'].iloc[0]}
GAME: {df['Application'].iloc[0]} | API: {df['Runtime'].iloc[0]} | VSync: {'ON' if df['SyncInterval'].iloc[0] == 1 else 'OFF'}

{full_data}

Provide DETAILED analysis covering:
1. BOTTLENECK ANALYSIS — GPU-bound or CPU-bound? Evidence?
2. THERMAL & POWER — Throttling? TDP concerns?
3. LATENCY — VSync impact, should they use Reflex?
4. FRAME PACING — Consistency, micro-stutters?
5. RESOLUTION SCALING — Is the performance drop normal?
6. TOP 5 OPTIMIZATION RECOMMENDATIONS — Specific settings to change
7. DLSS/FSR — Would it help? Expected FPS gain?
8. VERDICT — Rate each resolution (Excellent/Good/Playable/Poor)
9. UPGRADE PATH — Best single hardware upgrade?"""

        print("  [AI Advisor] Running detailed FrameView analysis...")
        analysis = self._query(prompt)

        return {
            "type": "frameview_detailed",
            "backend": self.backend,
            "model": self.model_name,
            "analysis": analysis,
            "reasoning": self.last_reasoning,
        }

    def analyze_cuda_results(self, cuda_json_path):
        """Analyze CUDA shader benchmark results (from cuda_bench binary output)."""
        path = Path(cuda_json_path)
        if not path.exists():
            return {"error": f"File not found: {path}"}

        with open(path) as f:
            data = json.load(f)

        gpu = data.get("gpu", "Unknown")
        vram = data.get("vram_mb", "?")
        results = data.get("results", [])

        # Format each kernel result into readable text
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
3. How does performance scale from 1080p to 4K?
4. Optimization recommendations for the worst kernel?
5. Overall: is this GPU suitable for 4K gaming?"""

        print("  [AI Advisor] Analyzing CUDA benchmark results...")
        analysis = self._query(prompt)

        return {
            "type": "cuda_analysis",
            "gpu": gpu,
            "backend": self.backend,
            "model": self.model_name,
            "analysis": analysis,
            "reasoning": self.last_reasoning,
        }

    def analyze_regressions(self, regressions):
        """Analyze performance regressions detected between benchmark runs."""
        if not regressions:
            return {"analysis": "No regressions detected. Performance is consistent."}

        # Format each regression into readable text
        reg_text = ""
        for r in regressions:
            if "kernel" in r:
                # CUDA kernel regression (cross-resolution)
                reg_text += (
                    f"  [{r['severity']}] {r['kernel']}: "
                    f"{r['from_res']}→{r['to_res']} "
                    f"+{r['change_pct']}% frame time increase\n"
                )
            else:
                # FrameView metric regression (cross-run)
                reg_text += (
                    f"  [{r.get('severity', '?')}] {r.get('metric', '?')}: "
                    f"{r.get('change_pct', 0):+.1f}% change\n"
                )

        prompt = f"""These GPU performance regressions were detected:

{reg_text}

For each regression:
1. Likely root cause?
2. Expected (e.g., higher resolution) or unexpected?
3. Hardware or software fix?
4. Priority ranking — which to investigate first?"""

        print("  [AI Advisor] Analyzing performance regressions...")
        analysis = self._query(prompt)

        return {
            "type": "regression_analysis",
            "num_regressions": len(regressions),
            "backend": self.backend,
            "analysis": analysis,
            "reasoning": self.last_reasoning,
        }


    # Full Report Generator — combines all analysis sections into one report

    def generate_full_report(self, cuda_json=None, fv_stats=None,
                              fv_dataframes=None, regressions=None):
        """
        Generate a comprehensive AI performance report combining all available data.

        Args:
            cuda_json:      Path to CUDA benchmark JSON (from cuda_bench)
            fv_stats:       List of stats dicts from FrameViewParser.analyze()
            fv_dataframes:  Dict of {"1080p": df, "1440p": df} for detailed analysis
            regressions:    List of regression dicts from RegressionDetector
        """
        print(f"\n{'='*60}")
        print(f"  [AI ADVISOR] Generating full performance report...")
        print(f"  Backend: {self.backend} ({self.model_name or 'rule-based'})")
        print(f"{'='*60}")

        sections = []

        # Detailed FrameView analysis (preferred — uses raw DataFrames with all columns)
        if fv_dataframes:
            report = self.analyze_frameview_detailed(fv_dataframes)
            sections.append(("FrameView Detailed Analysis", report["analysis"]))

        # Basic FrameView analysis (fallback — uses pre-parsed stats dicts)
        elif fv_stats:
            report = self.analyze_frameview_results(fv_stats)
            sections.append(("FrameView Performance Analysis", report["analysis"]))

        # CUDA shader benchmark analysis
        if cuda_json and Path(cuda_json).exists():
            report = self.analyze_cuda_results(cuda_json)
            sections.append(("CUDA Shader Benchmark Analysis", report["analysis"]))

        # Regression root cause analysis
        if regressions:
            report = self.analyze_regressions(regressions)
            sections.append(("Regression Root Cause Analysis", report["analysis"]))

        # Build the final formatted report
        backend_label = {
            "deepseek": f"DeepSeek {'R1' if self.use_reasoning else 'V3'} ({self.model_name})",
            "nvidia": f"NVIDIA NIM ({self.model_name})",
            "fallback": "Rule-Based Fallback",
        }.get(self.backend, "Unknown")

        full_report = "=" * 60 + "\n"
        full_report += "  FrameForge AI Performance Report\n"
        full_report += f"  Powered by {backend_label}\n"
        full_report += "=" * 60 + "\n\n"

        for title, content in sections:
            full_report += f"## {title}\n\n{content}\n\n{'-' * 40}\n\n"

        return full_report

    def save_report(self, report, output_dir="results"):
        """Save the AI report and R1 reasoning chain to text files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main report
        report_path = output_dir / "ai_advisor_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Saved: {report_path}")

        # Save DeepSeek R1 reasoning chain if available
        if self.last_reasoning:
            reasoning_path = output_dir / "ai_advisor_reasoning.txt"
            with open(reasoning_path, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("  DeepSeek R1 — Internal Reasoning Chain\n")
                f.write("  (Shows how R1 analyzed your benchmark data)\n")
                f.write("=" * 60 + "\n\n")
                f.write(self.last_reasoning)
            print(f"  Saved: {reasoning_path}")

        return report_path


# Standalone usage — run directly from command line

if __name__ == "__main__":
    import sys

    advisor = PerformanceAdvisor()

    # Exit early if no backend is available
    if not advisor.available:
        print("\nTo enable AI analysis, set an API key:")
        print("  export DEEPSEEK_API_KEY=sk-XXXXX   (free at platform.deepseek.com)")
        print("  export NVIDIA_API_KEY=nvapi-XXXXX   (free at build.nvidia.com)")
        sys.exit(1)

    # If a CUDA JSON path is provided, analyze it
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        report = advisor.analyze_cuda_results(sys.argv[1])
        print(report["analysis"])
    else:
        print("Usage:")
        print("  python ai_advisor.py <cuda_results.json>")
        print("  Or import and use in notebook — see docstring for examples")