# ONNX Model Optimization Pipeline

Convert, optimize, and benchmark PyTorch models using ONNX Runtime, with graph optimization, quantization and dynamic batching. Supports vision and NLP models. (TensorRT is not implemented; the benchmark uses the ONNX Runtime execution providers set in `configs/config.yaml`, CPU by default.)

## Overview

This pipeline provides an end-to-end workflow for taking trained PyTorch models through ONNX conversion, applying multiple optimization strategies, benchmarking performance gains, and serving optimized models through a FastAPI inference server.

## Features

- **Model Conversion**: Export PyTorch and Hugging Face Transformer models to ONNX format with proper dynamic axes and opset configuration
- **Graph Optimization**: Constant folding, operator fusion, and dead code elimination via ONNX Runtime graph optimizers
- **Quantization**: Dynamic INT8 quantization, static quantization with calibration datasets, and mixed-precision FP16 for GPU
- **Benchmarking**: Comprehensive latency (p50/p95/p99), throughput, and memory profiling across batch sizes
- **Inference Server**: FastAPI-based serving with dynamic batching, health checks, and automatic execution provider selection
- **Visualization**: Latency comparison plots and throughput scaling charts

## Quick Start

```bash
pip install -r requirements.txt

# Convert a PyTorch model to ONNX
python -c "
from src.converter import ModelConverter
from models.example_model import create_example_model

model, dummy_input = create_example_model()
converter = ModelConverter()
converter.export_pytorch_to_onnx(model, dummy_input, 'output/model.onnx')
converter.validate_onnx_model('output/model.onnx', dummy_input)
"

# Optimize the model
python -c "
from src.optimizer import ModelOptimizer
optimizer = ModelOptimizer('output/model.onnx')
optimizer.apply_all_optimizations('output/model_optimized.onnx')
optimizer.get_optimization_report()
"

# Benchmark
python -c "
from src.benchmark import ModelBenchmark
bench = ModelBenchmark()
bench.compare_models({
    'original': 'output/model.onnx',
    'optimized': 'output/model_optimized.onnx'
})
"

# Start inference server
uvicorn src.inference_server:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
onnx-model-optimization/
├── src/
│   ├── converter.py          # PyTorch to ONNX conversion
│   ├── optimizer.py           # Quantization and graph optimization
│   ├── benchmark.py           # Latency and throughput benchmarking
│   └── inference_server.py    # FastAPI inference server
├── models/
│   └── example_model.py       # Example PyTorch model for demos
├── configs/
│   └── config.yaml            # Pipeline configuration
├── requirements.txt
└── README.md
```

## Optimization Results

No benchmark results are published in this repository. The Benchmark step in Quick Start (`ModelBenchmark.compare_models`) measures p50/p95/p99 latency, throughput and memory for each model file on your hardware, and `generate_report` / `plot_latency_comparison` save the results. Speedups depend heavily on model, hardware and execution provider.

## Configuration

See `configs/config.yaml` for full configuration options including opset version, quantization type, benchmark parameters, and inference server settings.

## License

MIT License
