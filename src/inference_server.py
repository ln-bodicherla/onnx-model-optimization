"""
Optimized inference server for serving ONNX models via FastAPI.

Features dynamic batching, health checks, automatic execution provider
selection, and batch accumulation for maximizing throughput under load.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import numpy as np
import onnxruntime as ort
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Single prediction request payload."""
    inputs: list[list[float]] = Field(
        ...,
        description="Input tensor as nested list of floats",
        examples=[[[0.1, 0.2, 0.3]]],
    )
    output_names: Optional[list[str]] = Field(
        default=None,
        description="Specific output names to return",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    inputs: list[list[list[float]]] = Field(
        ...,
        description="List of input tensors",
    )


class PredictionResponse(BaseModel):
    """Prediction response payload."""
    outputs: list[list[float]]
    latency_ms: float
    model_name: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response payload."""
    outputs: list[list[list[float]]]
    total_latency_ms: float
    batch_size: int
    model_name: str


class ServerHealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str
    execution_providers: list[str]
    input_shape: list[Any]
    output_shape: list[Any]


class OptimizedInferenceServer:
    """High-performance ONNX model serving with dynamic batching.

    Manages model loading, session configuration, and provides both
    single and batch prediction interfaces. Supports dynamic batching
    to accumulate individual requests into efficient batches.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = self._load_config(config_path)
        self.session: Optional[ort.InferenceSession] = None
        self.model_path: Optional[str] = None
        self.model_name: str = "unknown"

        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._batch_size: int = 8
        self._batch_timeout: float = 0.05
        self._batch_task: Optional[asyncio.Task] = None

        inference_config = self.config.get("inference", {})
        self._providers = inference_config.get(
            "execution_providers",
            ["CPUExecutionProvider"],
        )
        self._inter_op_threads = inference_config.get("inter_op_threads", 4)
        self._intra_op_threads = inference_config.get("intra_op_threads", 4)

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load server configuration from YAML."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def load_model(self, model_path: str) -> None:
        """Load an ONNX model and create an optimized inference session.

        Automatically selects available execution providers and configures
        thread pools for optimal performance.

        Args:
            model_path: Path to the ONNX model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]

        available = ort.get_available_providers()
        providers = [p for p in self._providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = self._inter_op_threads
        sess_options.intra_op_num_threads = self._intra_op_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=providers,
        )

        logger.info(
            "Model loaded: %s (providers: %s)",
            self.model_name,
            providers,
        )

    def setup_dynamic_batching(
        self,
        max_batch_size: int = 8,
        timeout_ms: float = 50.0,
    ) -> None:
        """Configure dynamic batching parameters.

        Dynamic batching accumulates individual prediction requests into
        batches for more efficient GPU/CPU utilization.

        Args:
            max_batch_size: Maximum requests per batch.
            timeout_ms: Max wait time before processing partial batch.
        """
        self._batch_size = max_batch_size
        self._batch_timeout = timeout_ms / 1000.0
        logger.info(
            "Dynamic batching configured: max_batch=%d, timeout=%.1fms",
            max_batch_size,
            timeout_ms,
        )

    async def _process_batch_queue(self) -> None:
        """Background task that processes the batch accumulation queue."""
        while True:
            batch_items = []
            try:
                first_item = await self._batch_queue.get()
                batch_items.append(first_item)

                deadline = time.monotonic() + self._batch_timeout
                while len(batch_items) < self._batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._batch_queue.get(),
                            timeout=remaining,
                        )
                        batch_items.append(item)
                    except asyncio.TimeoutError:
                        break

                inputs = np.concatenate(
                    [item["input"] for item in batch_items],
                    axis=0,
                )
                results = self._run_inference(inputs)

                for i, item in enumerate(batch_items):
                    item["future"].set_result(results[i:i + 1])

            except Exception as e:
                for item in batch_items:
                    if not item["future"].done():
                        item["future"].set_exception(e)

    def _run_inference(self, input_array: np.ndarray) -> np.ndarray:
        """Execute inference on the loaded model.

        Args:
            input_array: Input tensor as numpy array.

        Returns:
            Model output as numpy array.
        """
        if self.session is None:
            raise RuntimeError("No model loaded")

        input_name = self.session.get_inputs()[0].name
        feed = {input_name: input_array.astype(np.float32)}
        outputs = self.session.run(None, feed)
        return outputs[0]

    def predict(self, inputs: np.ndarray) -> dict[str, Any]:
        """Run single prediction synchronously.

        Args:
            inputs: Input array with shape matching model expectations.

        Returns:
            Prediction results with outputs and latency.
        """
        start = time.perf_counter()
        outputs = self._run_inference(inputs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "outputs": outputs.tolist(),
            "latency_ms": round(elapsed_ms, 3),
            "model_name": self.model_name,
        }

    def predict_batch(
        self,
        inputs_list: list[np.ndarray],
        chunk_size: int = 32,
    ) -> dict[str, Any]:
        """Run batch prediction with optional chunking.

        Args:
            inputs_list: List of input arrays.
            chunk_size: Maximum items per inference call.

        Returns:
            Batch prediction results with all outputs and total latency.
        """
        start = time.perf_counter()
        all_outputs = []

        batched = np.concatenate(inputs_list, axis=0)

        for i in range(0, len(batched), chunk_size):
            chunk = batched[i:i + chunk_size]
            result = self._run_inference(chunk)
            all_outputs.append(result)

        combined = np.concatenate(all_outputs, axis=0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "outputs": combined.tolist(),
            "total_latency_ms": round(elapsed_ms, 3),
            "batch_size": len(batched),
            "model_name": self.model_name,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        if self.session is None:
            return {"error": "No model loaded"}

        inputs = []
        for inp in self.session.get_inputs():
            inputs.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            })

        outputs = []
        for out in self.session.get_outputs():
            outputs.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })

        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "providers": self.session.get_providers(),
            "inputs": inputs,
            "outputs": outputs,
        }


server = OptimizedInferenceServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    model_path = os.environ.get("MODEL_PATH", "models/model.onnx")
    if os.path.exists(model_path):
        server.load_model(model_path)
        server.setup_dynamic_batching()
        logger.info("Inference server ready")
    else:
        logger.warning("Model not found at %s, server started without model", model_path)
    yield
    logger.info("Inference server shutting down")


app = FastAPI(
    title="ONNX Inference Server",
    description="Optimized model serving with dynamic batching",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run single prediction."""
    if server.session is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    inputs = np.array(request.inputs, dtype=np.float32)
    result = server.predict(inputs)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Run batch prediction."""
    if server.session is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    inputs_list = [np.array(inp, dtype=np.float32) for inp in request.inputs]
    result = server.predict_batch(inputs_list)
    return BatchPredictionResponse(**result)


@app.get("/health", response_model=ServerHealthResponse)
async def health() -> ServerHealthResponse:
    """Health check endpoint."""
    if server.session is None:
        return ServerHealthResponse(
            status="degraded",
            model_loaded=False,
            model_path="",
            execution_providers=[],
            input_shape=[],
            output_shape=[],
        )

    info = server.get_model_info()
    return ServerHealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=info["model_path"],
        execution_providers=info["providers"],
        input_shape=[i["shape"] for i in info["inputs"]],
        output_shape=[o["shape"] for o in info["outputs"]],
    )


@app.get("/model/info")
async def model_info():
    """Return model metadata."""
    if server.session is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return server.get_model_info()
