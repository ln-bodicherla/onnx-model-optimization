"""
Model converter for exporting PyTorch models to ONNX format.

Handles standard PyTorch models and Hugging Face Transformer models with
proper dynamic axes, opset versioning, validation, and numerical accuracy
comparison.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


class ModelConverter:
    """Converts PyTorch models to ONNX format with validation and optimization.

    Supports both standard PyTorch nn.Module models and Hugging Face
    Transformer models. Handles dynamic axes configuration, opset version
    selection, and post-export validation.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.opset_version = self.config.get("conversion", {}).get("opset_version", 17)
        self.dynamic_axes = self.config.get("conversion", {}).get("dynamic_axes", {})

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Config not found at %s, using defaults", config_path)
            return {}

    def export_pytorch_to_onnx(
        self,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, tuple[torch.Tensor, ...]],
        output_path: str,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
        opset_version: Optional[int] = None,
    ) -> str:
        """Export a PyTorch model to ONNX format.

        Args:
            model: Trained PyTorch model in eval mode.
            dummy_input: Example input tensor(s) for tracing.
            output_path: Path to save the ONNX model.
            input_names: Names for input nodes. Defaults to ["input"].
            output_names: Names for output nodes. Defaults to ["output"].
            dynamic_axes: Dynamic axis configuration for variable-size dimensions.
            opset_version: ONNX opset version. Uses config default if not specified.

        Returns:
            Absolute path to the exported ONNX model file.
        """
        model.eval()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        input_names = input_names or ["input"]
        output_names = output_names or ["output"]
        opset_version = opset_version or self.opset_version
        dynamic_axes = dynamic_axes or self.dynamic_axes or {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

        logger.info(
            "Exporting PyTorch model to ONNX (opset=%d, output=%s)",
            opset_version,
            output_path,
        )

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Model exported successfully (%.2f MB)", file_size_mb)

        return os.path.abspath(output_path)

    def export_transformers_to_onnx(
        self,
        model_name_or_path: str,
        output_path: str,
        task: str = "text-classification",
        max_length: int = 128,
        opset_version: Optional[int] = None,
    ) -> str:
        """Export a Hugging Face Transformer model to ONNX format.

        Handles the special input structure (input_ids, attention_mask, etc.)
        and sets up appropriate dynamic axes for sequence length and batch size.

        Args:
            model_name_or_path: HF model identifier or local path.
            output_path: Path to save the ONNX model.
            task: Model task type for determining inputs/outputs.
            max_length: Maximum sequence length for dummy inputs.
            opset_version: ONNX opset version override.

        Returns:
            Absolute path to the exported ONNX model file.
        """
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading transformer model: %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        model.eval()

        dummy_text = "This is a sample input for ONNX export validation."
        encoded = tokenizer(
            dummy_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_names = list(encoded.keys())
        output_names = ["last_hidden_state"]
        dynamic_axes = {}
        for name in input_names:
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}

        dummy_input = tuple(encoded.values())
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version or self.opset_version,
                do_constant_folding=True,
                export_params=True,
            )

        logger.info("Transformer model exported to %s", output_path)
        return os.path.abspath(output_path)

    def validate_onnx_model(
        self,
        onnx_path: str,
        dummy_input: Optional[Union[torch.Tensor, dict[str, np.ndarray]]] = None,
    ) -> dict[str, Any]:
        """Validate an ONNX model's structure and run test inference.

        Checks the model graph for correctness using onnx.checker, then
        optionally runs inference with a dummy input to verify the model
        produces outputs.

        Args:
            onnx_path: Path to the ONNX model file.
            dummy_input: Optional test input for inference validation.

        Returns:
            Validation report with structure and inference results.
        """
        logger.info("Validating ONNX model: %s", onnx_path)
        report = {
            "path": onnx_path,
            "valid_structure": False,
            "inference_ok": False,
            "errors": [],
        }

        try:
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model, full_check=True)
            report["valid_structure"] = True
            report["opset_version"] = model.opset_import[0].version
            report["ir_version"] = model.ir_version
            logger.info("Model structure validation passed")
        except Exception as e:
            report["errors"].append(f"Structure validation failed: {str(e)}")
            logger.error("Structure validation failed: %s", e)
            return report

        if dummy_input is not None:
            try:
                session = ort.InferenceSession(
                    onnx_path,
                    providers=["CPUExecutionProvider"],
                )

                if isinstance(dummy_input, torch.Tensor):
                    input_name = session.get_inputs()[0].name
                    feed = {input_name: dummy_input.numpy()}
                elif isinstance(dummy_input, dict):
                    feed = dummy_input
                else:
                    input_name = session.get_inputs()[0].name
                    feed = {input_name: np.array(dummy_input)}

                outputs = session.run(None, feed)
                report["inference_ok"] = True
                report["output_shapes"] = [list(o.shape) for o in outputs]
                logger.info("Inference validation passed, output shapes: %s", report["output_shapes"])

            except Exception as e:
                report["errors"].append(f"Inference validation failed: {str(e)}")
                logger.error("Inference validation failed: %s", e)

        return report

    def simplify_model(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Run ONNX graph simplification to optimize the model.

        Performs constant folding, dead node elimination, and other
        graph-level simplifications that reduce model complexity without
        affecting accuracy.

        Args:
            input_path: Path to the input ONNX model.
            output_path: Path to save simplified model. Defaults to input with _simplified suffix.

        Returns:
            Path to the simplified model.
        """
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_simplified{ext}"

        logger.info("Simplifying ONNX model: %s", input_path)
        model = onnx.load(input_path)

        try:
            from onnxsim import simplify
            simplified_model, check = simplify(model)
            if not check:
                logger.warning("Simplified model failed validation check")
            onnx.save(simplified_model, output_path)
        except ImportError:
            logger.info("onnxsim not available, applying basic optimizations")
            from onnx import optimizer as onnx_optimizer
            passes = [
                "eliminate_identity",
                "eliminate_nop_pad",
                "eliminate_nop_transpose",
                "eliminate_unused_initializer",
                "fuse_bn_into_conv",
                "fuse_consecutive_squeezes",
                "fuse_consecutive_transposes",
            ]
            optimized = onnx_optimizer.optimize(model, passes)
            onnx.save(optimized, output_path)

        original_size = os.path.getsize(input_path) / (1024 * 1024)
        simplified_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(
            "Simplification complete: %.2f MB -> %.2f MB (%.1f%% reduction)",
            original_size,
            simplified_size,
            (1 - simplified_size / original_size) * 100 if original_size > 0 else 0,
        )

        return output_path

    def compare_outputs(
        self,
        pytorch_model: nn.Module,
        onnx_path: str,
        test_input: torch.Tensor,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> dict[str, Any]:
        """Compare PyTorch and ONNX model outputs for numerical accuracy.

        Runs the same input through both models and compares outputs
        using numpy's allclose with configurable tolerances.

        Args:
            pytorch_model: Original PyTorch model.
            onnx_path: Path to the exported ONNX model.
            test_input: Test input tensor.
            atol: Absolute tolerance for comparison.
            rtol: Relative tolerance for comparison.

        Returns:
            Comparison report with max difference and pass/fail status.
        """
        pytorch_model.eval()

        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
            if isinstance(pytorch_output, tuple):
                pytorch_output = pytorch_output[0]
            pytorch_np = pytorch_output.numpy()

        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: test_input.numpy()})[0]

        max_abs_diff = float(np.max(np.abs(pytorch_np - onnx_output)))
        mean_abs_diff = float(np.mean(np.abs(pytorch_np - onnx_output)))
        matches = bool(np.allclose(pytorch_np, onnx_output, atol=atol, rtol=rtol))

        report = {
            "matches": matches,
            "max_absolute_difference": max_abs_diff,
            "mean_absolute_difference": mean_abs_diff,
            "atol": atol,
            "rtol": rtol,
            "pytorch_output_shape": list(pytorch_np.shape),
            "onnx_output_shape": list(onnx_output.shape),
        }

        if matches:
            logger.info("Output comparison PASSED (max diff: %.2e)", max_abs_diff)
        else:
            logger.warning("Output comparison FAILED (max diff: %.2e)", max_abs_diff)

        return report

    def get_model_info(self, onnx_path: str) -> dict[str, Any]:
        """Extract detailed information about an ONNX model.

        Returns model size, operator counts, input/output specifications,
        and metadata.

        Args:
            onnx_path: Path to the ONNX model file.

        Returns:
            Dictionary with comprehensive model information.
        """
        model = onnx.load(onnx_path)
        graph = model.graph

        op_counts: dict[str, int] = {}
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        inputs = []
        for inp in graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            inputs.append({
                "name": inp.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
            })

        outputs = []
        for out in graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            outputs.append({
                "name": out.name,
                "shape": shape,
                "dtype": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
            })

        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

        return {
            "path": onnx_path,
            "file_size_mb": round(file_size_mb, 2),
            "opset_version": model.opset_import[0].version,
            "ir_version": model.ir_version,
            "producer": model.producer_name,
            "total_nodes": len(graph.node),
            "total_parameters": sum(
                int(np.prod(
                    [d.dim_value for d in init.dims]
                    if hasattr(init, "dims")
                    else [init.dims[i] for i in range(len(init.dims))]
                ))
                for init in graph.initializer
            ),
            "op_counts": dict(sorted(op_counts.items(), key=lambda x: -x[1])),
            "inputs": inputs,
            "outputs": outputs,
        }
