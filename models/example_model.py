"""
Example PyTorch model for demonstrating the ONNX optimization pipeline.

Provides a ResNet-based image classifier and a simple MLP classifier,
along with factory functions for creating models with dummy inputs
suitable for export and benchmarking.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection.

    Implements a two-layer convolution with batch normalization and
    an optional projection shortcut for dimension matching.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class MiniResNet(nn.Module):
    """Lightweight ResNet-based image classifier.

    A compact version of ResNet suitable for demonstration and benchmarking
    of the ONNX optimization pipeline. Configurable depth and width.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        base_width: int = 64,
        layers: tuple[int, ...] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(
            in_channels, base_width, kernel_size=7,
            stride=2, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_width, layers[0])
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, num_classes)

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a layer consisting of multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, out_channels, kernel_size=1,
                    stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            ResidualBlock(self.in_planes, out_channels, stride, downsample),
        ]
        self.in_planes = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """Apply Kaiming initialization to convolution layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for tabular data.

    Useful for demonstrating ONNX export and optimization with
    simpler, non-convolutional architectures.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        output_dim: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_example_model(
    model_type: str = "resnet",
    num_classes: int = 10,
    image_size: int = 224,
    batch_size: int = 1,
) -> tuple[nn.Module, torch.Tensor]:
    """Factory function to create an example model with matching dummy input.

    Args:
        model_type: Type of model ("resnet" or "mlp").
        num_classes: Number of output classes.
        image_size: Input image dimensions (for resnet).
        batch_size: Batch size for the dummy input.

    Returns:
        Tuple of (model, dummy_input) ready for ONNX export.
    """
    if model_type == "resnet":
        model = MiniResNet(
            num_classes=num_classes,
            in_channels=3,
            base_width=64,
            layers=(2, 2, 2, 2),
        )
        model.eval()
        dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    elif model_type == "mlp":
        input_dim = 128
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=(256, 128, 64),
            output_dim=num_classes,
            dropout=0.0,
        )
        model.eval()
        dummy_input = torch.randn(batch_size, input_dim)

    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'resnet' or 'mlp'.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model_type}")
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Input shape: {list(dummy_input.shape)}")

    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {list(output.shape)}")

    return model, dummy_input


def create_calibration_data(
    model_type: str = "resnet",
    num_samples: int = 100,
    image_size: int = 224,
    input_dim: int = 128,
) -> list[dict[str, object]]:
    """Generate calibration data for static quantization.

    Creates random inputs in the format expected by the calibration
    data reader used during static quantization.

    Args:
        model_type: Type of model for input shape determination.
        num_samples: Number of calibration samples.
        image_size: Image dimensions (for resnet).
        input_dim: Feature dimensions (for mlp).

    Returns:
        List of dictionaries mapping input name to numpy arrays.
    """
    import numpy as np

    data = []
    for _ in range(num_samples):
        if model_type == "resnet":
            sample = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        else:
            sample = np.random.randn(1, input_dim).astype(np.float32)

        data.append({"input": sample})

    return data


if __name__ == "__main__":
    print("=== ResNet Model ===")
    resnet_model, resnet_input = create_example_model("resnet")

    print("\n=== MLP Model ===")
    mlp_model, mlp_input = create_example_model("mlp")
