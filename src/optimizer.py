"""
Model optimizer for applying quantization, graph optimization, and
mixed-precision transformations to ONNX models.

Supports dynamic and static INT8 quantization, FP16 mixed precision,
graph-level operator fusion, and model pruning.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import numpy_helper

logger = logging.getLogger(__name__)


class CalibrationDataReader:
    """Provides calibration data for static quantization.

    Iterates over a set of representative input samples that are used
    to determine optimal quantization ranges for activations.
    """

    def __init__(
        self,
        calibration_data: list[dict[str, np.ndarray]],
    ):
        self.data = calibration_data
        self.index = 0

    def get_next(self) -> Optional[dict[str, np.ndarray]]:
        """Return the next calibration sample, or None when exhausted."""
        if self.index >= len(self.data):
            return None
        sample = self.data[self.index]
        self.index += 1
        return sample

    def rewind(self) -> None:
        """Reset the reader to the beginning of the calibration set."""
        self.index = 0


class ModelOptimizer:
    """Applies various optimization techniques to ONNX models.

    Supports quantization (dynamic and static), graph optimization,
    mixed-precision FP16 conversion, and unused node pruning.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/config.yaml",
    ):
        self.model_path = os.path.abspath(model_path)
        self.config = self._load_config(config_path)
        self.optimization_log: list[dict[str, Any]] = []

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._original_size = os.path.getsize(self.model_path)
        logger.info(
            "ModelOptimizer initialized for %s (%.2f MB)",
            self.model_path,
            self._original_size / (1024 * 1024),
        )

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load optimization configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def quantize_dynamic(
        self,
        output_path: str,
        weight_type: str = "INT8",
        operators_to_quantize: Optional[list[str]] = None,
    ) -> str:
        """Apply dynamic quantization to the model.

        Dynamic quantization quantizes weights at export time and quantizes
        activations dynamically during inference. This provides good speedup
        with minimal accuracy loss and no calibration data requirement.

        Args:
            output_path: Path to save the quantized model.
            weight_type: Quantization type for weights ("INT8" or "UINT8").
            operators_to_quantize: Specific operators to quantize.

        Returns:
            Path to the quantized model.
        """
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quant_type_map = {
            "INT8": QuantType.QInt8,
            "UINT8": QuantType.QUInt8,
        }
        quant_type = quant_type_map.get(weight_type.upper(), QuantType.QInt8)

        if operators_to_quantize is None:
            operators_to_quantize = ["MatMul", "Attention", "Gemm", "Conv"]

        logger.info(
            "Applying dynamic quantization (weight_type=%s, ops=%s)",
            weight_type,
            operators_to_quantize,
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        quantize_dynamic(
            model_input=self.model_path,
            model_output=output_path,
            weight_type=quant_type,
            op_types_to_quantize=operators_to_quantize,
        )

        new_size = os.path.getsize(output_path)
        self._log_optimization(
            "dynamic_quantization",
            output_path,
            new_size,
            {"weight_type": weight_type},
        )

        return os.path.abspath(output_path)

    def quantize_static(
        self,
        output_path: str,
        calibration_data: list[dict[str, np.ndarray]],
        quant_format: str = "QDQ",
        activation_type: str = "INT8",
        weight_type: str = "INT8",
    ) -> str:
        """Apply static quantization using calibration data.

        Static quantization pre-computes quantization ranges for both
        weights and activations using representative calibration data.
        This typically achieves better performance than dynamic quantization.

        Args:
            output_path: Path to save the quantized model.
            calibration_data: List of input dictionaries for calibration.
            quant_format: Quantization format ("QOperator" or "QDQ").
            activation_type: Quantization type for activations.
            weight_type: Quantization type for weights.

        Returns:
            Path to the quantized model.
        """
        from onnxruntime.quantization import (
            quantize_static as ort_quantize_static,
            CalibrationMethod,
            QuantFormat,
            QuantType,
        )

        format_map = {
            "QOperator": QuantFormat.QOperator,
            "QDQ": QuantFormat.QDQ,
        }
        type_map = {
            "INT8": QuantType.QInt8,
            "UINT8": QuantType.QUInt8,
        }

        logger.info(
            "Applying static quantization (%s, calibration_samples=%d)",
            quant_format,
            len(calibration_data),
        )

        reader = CalibrationDataReader(calibration_data)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        ort_quantize_static(
            model_input=self.model_path,
            model_output=output_path,
            calibration_data_reader=reader,
            quant_format=format_map.get(quant_format, QuantFormat.QDQ),
            activation_type=type_map.get(activation_type.upper(), QuantType.QInt8),
            weight_type=type_map.get(weight_type.upper(), QuantType.QInt8),
            calibrate_method=CalibrationMethod.MinMax,
        )

        new_size = os.path.getsize(output_path)
        self._log_optimization(
            "static_quantization",
            output_path,
            new_size,
            {"format": quant_format, "samples": len(calibration_data)},
        )

        return os.path.abspath(output_path)

    def optimize_graph(
        self,
        output_path: str,
        optimization_level: str = "ORT_ENABLE_ALL",
    ) -> str:
        """Apply graph-level optimizations using ONNX Runtime.

        Performs constant folding, operator fusion (Conv+BN, MatMul+Add, etc.),
        dead code elimination, and other graph transformations.

        Args:
            output_path: Path to save the optimized model.
            optimization_level: ORT optimization level.

        Returns:
            Path to the optimized model.
        """
        level_map = {
            "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }

        opt_level = level_map.get(optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)

        logger.info("Applying graph optimizations (level=%s)", optimization_level)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = opt_level
        sess_options.optimized_model_filepath = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        _ = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        new_size = os.path.getsize(output_path)
        self._log_optimization(
            "graph_optimization",
            output_path,
            new_size,
            {"level": optimization_level},
        )

        return os.path.abspath(output_path)

    def mixed_precision(
        self,
        output_path: str,
        target_dtype: str = "float16",
        keep_io_types: bool = True,
    ) -> str:
        """Convert model to mixed precision (FP16) for GPU acceleration.

        Converts internal computations to FP16 while optionally keeping
        input/output types as FP32 for compatibility.

        Args:
            output_path: Path to save the FP16 model.
            target_dtype: Target precision ("float16").
            keep_io_types: Whether to keep I/O as FP32.

        Returns:
            Path to the mixed-precision model.
        """
        from onnxruntime.transformers import float16

        logger.info("Converting to mixed precision (%s)", target_dtype)

        model = onnx.load(self.model_path)

        if target_dtype == "float16":
            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=keep_io_types,
                min_positive_val=1e-7,
                max_finite_val=1e4,
            )
        else:
            raise ValueError(f"Unsupported target dtype: {target_dtype}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        onnx.save(model_fp16, output_path)

        new_size = os.path.getsize(output_path)
        self._log_optimization(
            "mixed_precision",
            output_path,
            new_size,
            {"dtype": target_dtype, "keep_io": keep_io_types},
        )

        return os.path.abspath(output_path)

    def prune_model(
        self,
        output_path: str,
        output_names: Optional[list[str]] = None,
    ) -> str:
        """Remove unused nodes from the model graph.

        Traces from specified outputs backward through the graph and
        removes any nodes that don't contribute to the final output.

        Args:
            output_path: Path to save the pruned model.
            output_names: Output node names to keep. Defaults to all outputs.

        Returns:
            Path to the pruned model.
        """
        logger.info("Pruning unused nodes from model")

        model = onnx.load(self.model_path)
        graph = model.graph

        if output_names is None:
            output_names = [o.name for o in graph.output]

        required_outputs = set(output_names)
        required_nodes = set()
        required_inputs = set()

        node_output_map = {}
        for node in graph.node:
            for output in node.output:
                node_output_map[output] = node

        def _trace_dependencies(output_name: str) -> None:
            """Recursively trace which nodes are needed for an output."""
            if output_name in node_output_map:
                node = node_output_map[output_name]
                node_key = (node.op_type, tuple(node.output))
                if node_key not in required_nodes:
                    required_nodes.add(node_key)
                    for inp in node.input:
                        _trace_dependencies(inp)
            else:
                required_inputs.add(output_name)

        for out_name in required_outputs:
            _trace_dependencies(out_name)

        original_count = len(graph.node)
        nodes_to_keep = []
        for node in graph.node:
            node_key = (node.op_type, tuple(node.output))
            if node_key in required_nodes:
                nodes_to_keep.append(node)

        del graph.node[:]
        graph.node.extend(nodes_to_keep)

        pruned_count = original_count - len(nodes_to_keep)
        logger.info("Pruned %d unused nodes (kept %d)", pruned_count, len(nodes_to_keep))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        onnx.save(model, output_path)

        new_size = os.path.getsize(output_path)
        self._log_optimization(
            "pruning",
            output_path,
            new_size,
            {"pruned_nodes": pruned_count, "remaining_nodes": len(nodes_to_keep)},
        )

        return os.path.abspath(output_path)

    def apply_all_optimizations(
        self,
        output_path: str,
        calibration_data: Optional[list[dict[str, np.ndarray]]] = None,
    ) -> str:
        """Apply a chain of optimizations in sequence.

        Applies graph optimization, then quantization (static if calibration
        data provided, dynamic otherwise).

        Args:
            output_path: Final output path for the fully optimized model.
            calibration_data: Optional calibration data for static quantization.

        Returns:
            Path to the fully optimized model.
        """
        logger.info("Applying full optimization pipeline")

        with tempfile.TemporaryDirectory(prefix="onnx_opt_") as tmpdir:
            graph_opt_path = os.path.join(tmpdir, "graph_optimized.onnx")
            self.optimize_graph(graph_opt_path)

            pruned_path = os.path.join(tmpdir, "pruned.onnx")
            pruner = ModelOptimizer.__new__(ModelOptimizer)
            pruner.model_path = graph_opt_path
            pruner._original_size = self._original_size
            pruner.optimization_log = self.optimization_log
            pruner.config = self.config
            pruner.prune_model(pruned_path)

            quant_optimizer = ModelOptimizer.__new__(ModelOptimizer)
            quant_optimizer.model_path = pruned_path
            quant_optimizer._original_size = self._original_size
            quant_optimizer.optimization_log = self.optimization_log
            quant_optimizer.config = self.config

            quant_config = self.config.get("optimization", {}).get("quantization", {})
            quant_type = quant_config.get("type", "dynamic")

            if quant_type == "static" and calibration_data:
                quant_optimizer.quantize_static(output_path, calibration_data)
            else:
                weight_type = quant_config.get("weight_type", "INT8")
                quant_optimizer.quantize_dynamic(output_path, weight_type=weight_type)

        logger.info("Full optimization pipeline complete: %s", output_path)
        return os.path.abspath(output_path)

    def get_optimization_report(self) -> dict[str, Any]:
        """Generate a comprehensive before/after optimization comparison.

        Returns:
            Report with size comparisons, optimization steps, and summary.
        """
        original_mb = self._original_size / (1024 * 1024)

        report = {
            "original_model": {
                "path": self.model_path,
                "size_mb": round(original_mb, 2),
            },
            "optimizations": [],
            "summary": {},
        }

        for entry in self.optimization_log:
            opt_size_mb = entry["new_size"] / (1024 * 1024)
            reduction_pct = (1 - entry["new_size"] / self._original_size) * 100

            report["optimizations"].append({
                "type": entry["type"],
                "output_path": entry["output_path"],
                "size_mb": round(opt_size_mb, 2),
                "size_reduction_pct": round(reduction_pct, 1),
                "params": entry["params"],
            })

        if self.optimization_log:
            final = self.optimization_log[-1]
            final_mb = final["new_size"] / (1024 * 1024)
            report["summary"] = {
                "original_size_mb": round(original_mb, 2),
                "final_size_mb": round(final_mb, 2),
                "total_reduction_pct": round(
                    (1 - final["new_size"] / self._original_size) * 100, 1
                ),
                "optimization_steps": len(self.optimization_log),
            }

        for key, value in report.items():
            logger.info("Report [%s]: %s", key, value)

        return report

    def _log_optimization(
        self,
        opt_type: str,
        output_path: str,
        new_size: int,
        params: dict[str, Any],
    ) -> None:
        """Record an optimization step in the log."""
        reduction = (1 - new_size / self._original_size) * 100
        self.optimization_log.append({
            "type": opt_type,
            "output_path": output_path,
            "new_size": new_size,
            "params": params,
        })
        logger.info(
            "%s: %.2f MB -> %.2f MB (%.1f%% reduction)",
            opt_type,
            self._original_size / (1024 * 1024),
            new_size / (1024 * 1024),
            reduction,
        )
