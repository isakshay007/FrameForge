"""
FrameForge — pytorch_bench.py
===============================
PyTorch model inference benchmarking on NVIDIA GPUs.
Compares FP32 vs FP16 (mixed precision) and torch.compile() optimization.
Uses NVTX annotations for Nsight Systems profiling.

Usage:
    python pytorch_bench.py                    # Run all benchmarks
    python pytorch_bench.py --model resnet50   # Specific model
    python pytorch_bench.py --output results/  # Save results
"""

import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Try importing NVTX for profiling annotations
try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    # Create dummy decorator
    class _DummyNvtx:
        @staticmethod
        def annotate(msg="", color=None):
            class _ctx:
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _ctx()
    nvtx = _DummyNvtx()


# =============================================================================
# Model Definitions
# =============================================================================
def get_resnet50():
    """ResNet-50 — standard vision model benchmark."""
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=None)  # No pretrained weights needed for benchmarking
    dummy_input = torch.randn(1, 3, 224, 224)
    return model, dummy_input, "ResNet-50"


def get_simple_transformer():
    """Simple Transformer encoder — simulates NLP workload."""
    config = {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 2048,
    }
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=config["d_model"],
        nhead=config["nhead"],
        dim_feedforward=config["dim_feedforward"],
        batch_first=True,
    )
    model = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
    # Sequence length 128, batch size 1
    dummy_input = torch.randn(1, 128, config["d_model"])
    return model, dummy_input, "Transformer-Encoder-6L"


def get_conv_net():
    """Custom ConvNet — simulates lightweight gaming AI model."""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, 10),
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    return model, dummy_input, "ConvNet-4L"


MODELS = {
    "resnet50": get_resnet50,
    "transformer": get_simple_transformer,
    "convnet": get_conv_net,
}


# =============================================================================
# Benchmark Engine
# =============================================================================
class PyTorchBenchmark:
    """Benchmarks PyTorch model inference with multiple optimization modes."""

    WARMUP = 20
    ITERATIONS = 100
    BATCH_SIZES = [1, 4, 16]

    def __init__(self, device=None):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.results = []

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.gpu_name = props.name
            self.vram_mb = props.total_memory // (1024 * 1024)
            print(f"  GPU: {self.gpu_name} ({self.vram_mb} MB)")
        else:
            self.gpu_name = "CPU"
            self.vram_mb = 0
            print("  Running on CPU (no GPU detected)")

    def _measure_inference(self, model, dummy_input, num_iters):
        """Measure inference time using CUDA events (accurate GPU timing)."""
        if self.device.type == "cuda":
            # Use CUDA events for precise GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start_event.record()

            for _ in range(num_iters):
                with torch.no_grad():
                    _ = model(dummy_input)

            end_event.record()
            torch.cuda.synchronize()

            total_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            for _ in range(num_iters):
                with torch.no_grad():
                    _ = model(dummy_input)
            total_ms = (time.perf_counter() - t0) * 1000

        avg_ms = total_ms / num_iters
        return avg_ms

    def benchmark_model(self, model_name, batch_size=1):
        """Run full benchmark suite for a single model."""
        print(f"\n  Benchmarking: {model_name} (batch={batch_size})")

        # Get model
        model_fn = MODELS.get(model_name)
        if not model_fn:
            print(f"  Unknown model: {model_name}")
            return

        model, dummy_input, display_name = model_fn()

        # Adjust batch size
        if batch_size > 1:
            shape = list(dummy_input.shape)
            shape[0] = batch_size
            dummy_input = torch.randn(shape)

        model = model.to(self.device).eval()
        dummy_input = dummy_input.to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        # ---- MODE 1: FP32 (baseline) ----
        with nvtx.annotate(f"FP32 Benchmark: {display_name}", color="blue"):
            # Warmup
            with nvtx.annotate("FP32 Warmup"):
                self._measure_inference(model, dummy_input, self.WARMUP)

            # Benchmark
            with nvtx.annotate("FP32 Benchmark"):
                fp32_ms = self._measure_inference(model, dummy_input, self.ITERATIONS)

        # Memory usage
        if self.device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_mb = 0

        result_fp32 = {
            "model": display_name,
            "model_key": model_name,
            "params_M": round(num_params, 2),
            "batch_size": batch_size,
            "mode": "FP32",
            "avg_latency_ms": round(fp32_ms, 3),
            "throughput_fps": round(batch_size * 1000.0 / fp32_ms, 2),
            "gpu_mem_mb": round(mem_mb, 1),
            "device": self.gpu_name,
        }
        self.results.append(result_fp32)
        print(f"    FP32:          {fp32_ms:.2f}ms | {result_fp32['throughput_fps']:.1f} samples/s")

        # ---- MODE 2: FP16 Mixed Precision ----
        with nvtx.annotate(f"FP16 Benchmark: {display_name}", color="green"):
            with nvtx.annotate("FP16 Warmup"):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    self._measure_inference(model, dummy_input, self.WARMUP)

            with nvtx.annotate("FP16 Benchmark"):
                if self.device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    start.record()
                    for _ in range(self.ITERATIONS):
                        with torch.no_grad():
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                _ = model(dummy_input)
                    end.record()
                    torch.cuda.synchronize()
                    fp16_ms = start.elapsed_time(end) / self.ITERATIONS
                else:
                    t0 = time.perf_counter()
                    for _ in range(self.ITERATIONS):
                        with torch.no_grad():
                            with torch.autocast(device_type="cpu", dtype=torch.float16):
                                _ = model(dummy_input)
                    fp16_ms = ((time.perf_counter() - t0) * 1000) / self.ITERATIONS

        if self.device.type == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()

        speedup_fp16 = fp32_ms / fp16_ms if fp16_ms > 0 else 0

        result_fp16 = {
            "model": display_name,
            "model_key": model_name,
            "params_M": round(num_params, 2),
            "batch_size": batch_size,
            "mode": "FP16 (mixed precision)",
            "avg_latency_ms": round(fp16_ms, 3),
            "throughput_fps": round(batch_size * 1000.0 / fp16_ms, 2),
            "gpu_mem_mb": round(mem_mb, 1),
            "speedup_vs_fp32": round(speedup_fp16, 2),
            "device": self.gpu_name,
        }
        self.results.append(result_fp16)
        print(f"    FP16:          {fp16_ms:.2f}ms | {result_fp16['throughput_fps']:.1f} samples/s | {speedup_fp16:.2f}x speedup")

        # ---- MODE 3: torch.compile() ----
        compile_available = hasattr(torch, "compile")
        if compile_available:
            with nvtx.annotate(f"torch.compile Benchmark: {display_name}", color="red"):
                try:
                    compiled_model = torch.compile(model, mode="reduce-overhead")

                    # Warmup (compile happens on first call)
                    with nvtx.annotate("torch.compile Warmup + Compilation"):
                        self._measure_inference(compiled_model, dummy_input, self.WARMUP + 5)

                    # Benchmark
                    with nvtx.annotate("torch.compile Benchmark"):
                        compiled_ms = self._measure_inference(compiled_model, dummy_input, self.ITERATIONS)

                    speedup_compile = fp32_ms / compiled_ms if compiled_ms > 0 else 0

                    result_compiled = {
                        "model": display_name,
                        "model_key": model_name,
                        "params_M": round(num_params, 2),
                        "batch_size": batch_size,
                        "mode": "torch.compile()",
                        "avg_latency_ms": round(compiled_ms, 3),
                        "throughput_fps": round(batch_size * 1000.0 / compiled_ms, 2),
                        "speedup_vs_fp32": round(speedup_compile, 2),
                        "device": self.gpu_name,
                    }
                    self.results.append(result_compiled)
                    print(f"    torch.compile: {compiled_ms:.2f}ms | {result_compiled['throughput_fps']:.1f} samples/s | {speedup_compile:.2f}x speedup")

                except Exception as e:
                    print(f"    torch.compile: FAILED ({e})")
        else:
            print(f"    torch.compile: Not available (PyTorch < 2.0)")

        # Cleanup
        del model, dummy_input
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run_all(self):
        """Run benchmarks for all models and batch sizes."""
        print("\n" + "=" * 50)
        print("  PyTorch Inference Benchmarks")
        print("=" * 50)
        print(f"  Device: {self.gpu_name}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        print(f"  Warmup: {self.WARMUP} | Iterations: {self.ITERATIONS}")

        for model_name in MODELS:
            for batch_size in [1, 16]:
                self.benchmark_model(model_name, batch_size)

        return self.results

    def save_results(self, output_dir="results"):
        """Save benchmark results to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "device": self.gpu_name,
            "vram_mb": self.vram_mb,
            "pytorch_version": torch.__version__,
            "cuda_version": str(torch.version.cuda) if torch.cuda.is_available() else "N/A",
            "warmup_iters": self.WARMUP,
            "bench_iters": self.ITERATIONS,
            "results": self.results,
        }

        path = output_dir / "pytorch_results.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\n  ✓ Results saved: {path}")
        return path

    def print_summary(self):
        """Print a formatted summary table."""
        print(f"\n{'=' * 80}")
        print(f"  {'Model':<25} {'Mode':<20} {'Batch':<6} {'Latency':<12} {'Throughput':<15} {'Speedup'}")
        print(f"{'=' * 80}")

        for r in self.results:
            speedup = r.get("speedup_vs_fp32", "")
            speedup_str = f"{speedup:.2f}x" if speedup else "baseline"
            print(
                f"  {r['model']:<25} {r['mode']:<20} {r['batch_size']:<6} "
                f"{r['avg_latency_ms']:.2f}ms{'':<6} "
                f"{r['throughput_fps']:.1f} samp/s{'':<4} "
                f"{speedup_str}"
            )

        print(f"{'=' * 80}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FrameForge PyTorch GPU Benchmark")
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Specific model to benchmark")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--output", default="../results", help="Output directory")
    args = parser.parse_args()

    bench = PyTorchBenchmark()

    if args.model:
        batch = args.batch or 1
        bench.benchmark_model(args.model, batch)
    else:
        bench.run_all()

    bench.print_summary()
    bench.save_results(args.output)


if __name__ == "__main__":
    main()