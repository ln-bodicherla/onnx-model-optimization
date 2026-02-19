"""
Benchmarking utilities for ONNX model performance evaluation.

Measures latency (average, p50, p95, p99), throughput at various batch sizes,
memory consumption, and generates comparison reports with visualizations.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import yaml

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Comprehensive benchmarking suite for ONNX models.

    Supports latency profiling, throughput measurement, memory tracking,
    model comparison, and report generation with visualizations.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        bench_config = self.config.get("benchmark", {})
        self.warmup_runs = bench_config.get("warmup_runs", 50)
        self.num_runs = bench_config.get("num_runs", 1000)
        self.batch_sizes = bench_config.get("batch_sizes", [1, 4, 8, 16, 32])

        inference_config = self.config.get("inference", {})
        self.execution_providers = inference_config.get(
            "execution_providers",
            ["CPUExecutionProvider"],
        )
        self.inter_op_threads = inference_config.get("inter_op_threads", 4)
        self.intra_op_threads = inference_config.get("intra_op_threads", 4)

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load benchmark configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def setup_session(
        self,
        model_path: str,
        providers: Optional[list[str]] = None,
    ) -> ort.InferenceSession:
        """Create an ONNX Runtime InferenceSession with optimal settings.

        Args:
            model_path: Path to the ONNX model.
            providers: Execution providers to use. Falls back to CPU if GPU unavailable.

        Returns:
            Configured InferenceSession ready for inference.
        """
        providers = providers or self.execution_providers

        available = ort.get_available_providers()
        valid_providers = [p for p in providers if p in available]
        if not valid_providers:
            valid_providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = self.inter_op_threads
        sess_options.intra_op_num_threads = self.intra_op_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        logger.info("Creating session with providers: %s", valid_providers)
        session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=valid_providers,
        )

        return session

    def _generate_dummy_input(
        self,
        session: ort.InferenceSession,
        batch_size: int = 1,
    ) -> dict[str, np.ndarray]:
        """Generate dummy input matching the model's expected input shape."""
        feed = {}
        for inp in session.get_inputs():
            shape = list(inp.shape)
            for i, dim in enumerate(shape):
                if isinstance(dim, str) or dim is None:
                    shape[i] = batch_size if i == 0 else 128
                elif dim <= 0:
                    shape[i] = batch_size if i == 0 else 128

            dtype_map = {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(int64)": np.int64,
                "tensor(int32)": np.int32,
                "tensor(bool)": np.bool_,
            }
            dtype = dtype_map.get(inp.type, np.float32)

            if dtype in (np.int64, np.int32):
                feed[inp.name] = np.random.randint(0, 1000, size=shape).astype(dtype)
            elif dtype == np.bool_:
                feed[inp.name] = np.ones(shape, dtype=dtype)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(dtype)

        return feed

    def benchmark_latency(
        self,
        model_path: str,
        batch_size: int = 1,
        num_runs: Optional[int] = None,
        warmup_runs: Optional[int] = None,
    ) -> dict[str, float]:
        """Measure inference latency with percentile statistics.

        Performs warmup runs first, then measures inference time over
        multiple iterations to compute average, median, and tail latencies.

        Args:
            model_path: Path to the ONNX model.
            batch_size: Batch size for input tensors.
            num_runs: Number of measurement runs. Uses config default if None.
            warmup_runs: Number of warmup iterations. Uses config default if None.

        Returns:
            Dictionary with avg, p50, p95, p99, min, max latencies in milliseconds.
        """
        num_runs = num_runs or self.num_runs
        warmup_runs = warmup_runs or self.warmup_runs

        session = self.setup_session(model_path)
        feed = self._generate_dummy_input(session, batch_size)

        logger.info("Warming up (%d runs, batch_size=%d)...", warmup_runs, batch_size)
        for _ in range(warmup_runs):
            session.run(None, feed)

        latencies = []
        logger.info("Benchmarking latency (%d runs)...", num_runs)
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, feed)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies_arr = np.array(latencies)

        result = {
            "batch_size": batch_size,
            "num_runs": num_runs,
            "avg_ms": float(np.mean(latencies_arr)),
            "std_ms": float(np.std(latencies_arr)),
            "min_ms": float(np.min(latencies_arr)),
            "max_ms": float(np.max(latencies_arr)),
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
        }

        logger.info(
            "Latency (batch=%d): avg=%.2fms, p50=%.2fms, p95=%.2fms, p99=%.2fms",
            batch_size,
            result["avg_ms"],
            result["p50_ms"],
            result["p95_ms"],
            result["p99_ms"],
        )

        return result

    def benchmark_throughput(
        self,
        model_path: str,
        batch_sizes: Optional[list[int]] = None,
        duration_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Measure maximum throughput at different batch sizes.

        For each batch size, runs inference continuously for the specified
        duration and computes samples per second.

        Args:
            model_path: Path to the ONNX model.
            batch_sizes: List of batch sizes to test. Uses config default if None.
            duration_seconds: How long to run each batch size test.

        Returns:
            Throughput results for each batch size.
        """
        batch_sizes = batch_sizes or self.batch_sizes
        session = self.setup_session(model_path)

        results = {}
        for bs in batch_sizes:
            feed = self._generate_dummy_input(session, bs)

            for _ in range(min(self.warmup_runs, 20)):
                session.run(None, feed)

            iterations = 0
            start_time = time.perf_counter()
            deadline = start_time + duration_seconds

            while time.perf_counter() < deadline:
                session.run(None, feed)
                iterations += 1

            elapsed = time.perf_counter() - start_time
            samples_per_sec = (iterations * bs) / elapsed

            results[f"batch_{bs}"] = {
                "batch_size": bs,
                "iterations": iterations,
                "elapsed_seconds": round(elapsed, 2),
                "samples_per_second": round(samples_per_sec, 1),
                "batches_per_second": round(iterations / elapsed, 1),
            }

            logger.info(
                "Throughput (batch=%d): %.1f samples/sec, %.1f batches/sec",
                bs,
                samples_per_sec,
                iterations / elapsed,
            )

        return results

    def benchmark_memory(
        self,
        model_path: str,
        batch_sizes: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Measure CPU/GPU memory usage during inference.

        Uses process memory tracking to measure peak memory consumption
        at each batch size.

        Args:
            model_path: Path to the ONNX model.
            batch_sizes: List of batch sizes to test.

        Returns:
            Memory usage statistics for each batch size.
        """
        import psutil

        batch_sizes = batch_sizes or self.batch_sizes[:3]
        process = psutil.Process(os.getpid())
        results = {}

        for bs in batch_sizes:
            gc.collect()
            mem_before = process.memory_info().rss / (1024 * 1024)

            session = self.setup_session(model_path)
            feed = self._generate_dummy_input(session, bs)

            mem_after_load = process.memory_info().rss / (1024 * 1024)

            for _ in range(50):
                session.run(None, feed)

            mem_after_inference = process.memory_info().rss / (1024 * 1024)

            results[f"batch_{bs}"] = {
                "batch_size": bs,
                "baseline_mb": round(mem_before, 1),
                "after_load_mb": round(mem_after_load, 1),
                "after_inference_mb": round(mem_after_inference, 1),
                "model_memory_mb": round(mem_after_load - mem_before, 1),
                "inference_overhead_mb": round(mem_after_inference - mem_after_load, 1),
            }

            del session
            gc.collect()

            logger.info(
                "Memory (batch=%d): model=%.1fMB, inference_overhead=%.1fMB",
                bs,
                results[f"batch_{bs}"]["model_memory_mb"],
                results[f"batch_{bs}"]["inference_overhead_mb"],
            )

        return results

    def compare_models(
        self,
        model_paths: dict[str, str],
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """Run side-by-side benchmarks on multiple model versions.

        Args:
            model_paths: Mapping of model names to file paths.
            batch_size: Batch size for comparison.

        Returns:
            Comparison report with latency and throughput for each model.
        """
        results = {}

        for name, path in model_paths.items():
            logger.info("Benchmarking model: %s (%s)", name, path)
            file_size = os.path.getsize(path) / (1024 * 1024)

            latency = self.benchmark_latency(path, batch_size=batch_size)
            throughput = self.benchmark_throughput(
                path,
                batch_sizes=[batch_size],
                duration_seconds=5.0,
            )

            results[name] = {
                "path": path,
                "file_size_mb": round(file_size, 2),
                "latency": latency,
                "throughput": throughput.get(f"batch_{batch_size}", {}),
            }

        if len(results) >= 2:
            names = list(results.keys())
            baseline = results[names[0]]
            for name in names[1:]:
                other = results[name]
                speedup = baseline["latency"]["avg_ms"] / max(other["latency"]["avg_ms"], 0.001)
                size_reduction = (1 - other["file_size_mb"] / max(baseline["file_size_mb"], 0.001)) * 100
                results[name]["vs_baseline"] = {
                    "latency_speedup": round(speedup, 2),
                    "size_reduction_pct": round(size_reduction, 1),
                }

        return results

    def generate_report(
        self,
        model_paths: dict[str, str],
        output_dir: str = "benchmark_results",
    ) -> dict[str, Any]:
        """Generate a comprehensive benchmark report with charts.

        Args:
            model_paths: Models to benchmark.
            output_dir: Directory to save report artifacts.

        Returns:
            Complete benchmark report.
        """
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "models": {},
            "comparison": {},
        }

        for name, path in model_paths.items():
            logger.info("Full benchmark for: %s", name)
            latency_results = {}
            for bs in self.batch_sizes:
                latency_results[f"batch_{bs}"] = self.benchmark_latency(
                    path, batch_size=bs, num_runs=min(self.num_runs, 200),
                )

            throughput = self.benchmark_throughput(path, duration_seconds=5.0)

            report["models"][name] = {
                "path": path,
                "file_size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                "latency_by_batch": latency_results,
                "throughput": throughput,
            }

        report["comparison"] = self.compare_models(model_paths)

        self.plot_latency_comparison(report, output_dir)
        self.plot_throughput_scaling(report, output_dir)

        return report

    def plot_latency_comparison(
        self,
        report: dict[str, Any],
        output_dir: str = "benchmark_results",
    ) -> str:
        """Generate a bar chart comparing latency across models.

        Args:
            report: Benchmark report data.
            output_dir: Directory to save the chart.

        Returns:
            Path to the saved chart image.
        """
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        model_names = list(report.get("comparison", {}).keys())
        if not model_names:
            model_names = list(report.get("models", {}).keys())

        avg_latencies = []
        p95_latencies = []
        p99_latencies = []

        for name in model_names:
            if name in report.get("comparison", {}):
                data = report["comparison"][name]["latency"]
            elif name in report.get("models", {}):
                first_batch = list(report["models"][name]["latency_by_batch"].values())[0]
                data = first_batch
            else:
                continue
            avg_latencies.append(data.get("avg_ms", 0))
            p95_latencies.append(data.get("p95_ms", 0))
            p99_latencies.append(data.get("p99_ms", 0))

        x = np.arange(len(model_names))
        width = 0.25

        ax.bar(x - width, avg_latencies, width, label="Average", color="#2196F3")
        ax.bar(x, p95_latencies, width, label="P95", color="#FF9800")
        ax.bar(x + width, p99_latencies, width, label="P99", color="#F44336")

        ax.set_xlabel("Model")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "latency_comparison.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        logger.info("Latency comparison chart saved to %s", chart_path)
        return chart_path

    def plot_throughput_scaling(
        self,
        report: dict[str, Any],
        output_dir: str = "benchmark_results",
    ) -> str:
        """Generate a line chart showing throughput vs batch size.

        Args:
            report: Benchmark report data.
            output_dir: Directory to save the chart.

        Returns:
            Path to the saved chart image.
        """
        import matplotlib.pyplot as plt

        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

        for idx, (name, model_data) in enumerate(report.get("models", {}).items()):
            throughput = model_data.get("throughput", {})
            batch_sizes = []
            samples_per_sec = []

            for key, data in sorted(throughput.items()):
                batch_sizes.append(data["batch_size"])
                samples_per_sec.append(data["samples_per_second"])

            if batch_sizes:
                color = colors[idx % len(colors)]
                ax.plot(
                    batch_sizes,
                    samples_per_sec,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=name,
                    color=color,
                )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/sec)")
        ax.set_title("Throughput Scaling by Batch Size")
        ax.legend()
        ax.grid(alpha=0.3)

        if batch_sizes:
            ax.set_xticks(batch_sizes)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "throughput_scaling.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

        logger.info("Throughput scaling chart saved to %s", chart_path)
        return chart_path
