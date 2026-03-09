#!/usr/bin/env python3
"""
Export ml-sharp model to ONNX format for web inference.

This script converts the PyTorch ml-sharp model to ONNX format,
which can then be used with ONNX Runtime Web in the browser.

Usage:
    python export_to_onnx.py --checkpoint path/to/checkpoint.pt --output model.onnx

Requirements:
    pip install torch onnx onnxruntime
    pip install -e path/to/ml-sharp  # Install ml-sharp package

For licensing see accompanying LICENSE file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# These imports require ml-sharp to be installed
# pip install -e path/to/ml-sharp
try:
    from sharp.models import PredictorParams, create_predictor
    from sharp.models.predictor import RGBGaussianPredictor
    ML_SHARP_AVAILABLE = True
except ImportError:
    ML_SHARP_AVAILABLE = False
    print("Warning: ml-sharp not installed. Install with: pip install -e path/to/ml-sharp")

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Default model URL from ml-sharp
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

# Fixed input size for the model (ml-sharp uses 1536x1536 internally)
INTERNAL_SIZE = (1536, 1536)


class SharpONNXWrapper(nn.Module):
    """
    Wrapper module for ONNX export that handles the Gaussians3D output.
    
    The original model returns a Gaussians3D dataclass, but ONNX requires
    tensor outputs. This wrapper extracts the tensor components.
    """
    
    def __init__(self, predictor: nn.Module):
        super().__init__()
        self.predictor = predictor
    
    def forward(
        self,
        image: torch.Tensor,
        disparity_factor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns individual tensor outputs.
        
        Args:
            image: Input image tensor [B, 3, H, W] normalized to [0, 1]
            disparity_factor: Disparity factor tensor [B]
            
        Returns:
            Tuple of:
                - mean_vectors: Position of each Gaussian [B, N, 3]
                - quaternions: Rotation quaternion [B, N, 4]
                - singular_values: Scale of each Gaussian [B, N, 3]
                - opacities: Opacity values [B, N]
                - colors: RGB colors (from SH DC) [B, N, 3]
        """
        gaussians = self.predictor(image, disparity_factor, depth=None)
        
        return (
            gaussians.mean_vectors,
            gaussians.quaternions,
            gaussians.singular_values,
            gaussians.opacities,
            gaussians.colors,
        )


def load_checkpoint(checkpoint_path: Path | None) -> dict[str, Any]:
    """Load model checkpoint from file or download from URL."""
    if checkpoint_path is not None:
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    else:
        LOGGER.info(f"Downloading checkpoint from {DEFAULT_MODEL_URL}")
        return torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, 
            map_location="cpu",
            progress=True
        )


def export_to_onnx(
    checkpoint_path: Path | None,
    output_path: Path,
    opset_version: int = 18,
    simplify: bool = True,
) -> None:
    """
    Export ml-sharp model to ONNX format.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file, or None to download default
        output_path: Output path for ONNX model
        opset_version: ONNX opset version (default 18 for broad compatibility)
        simplify: Whether to simplify the ONNX model
    """
    if not ML_SHARP_AVAILABLE:
        raise RuntimeError(
            "ml-sharp package not found. Please install it first:\n"
            "  git clone https://github.com/apple/ml-sharp\n"
            "  cd ml-sharp && pip install -e ."
        )
    
    # Load checkpoint
    state_dict = load_checkpoint(checkpoint_path)
    
    # Create model
    LOGGER.info("Creating model...")
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    
    # Wrap for ONNX export
    wrapper = SharpONNXWrapper(predictor)
    wrapper.eval()
    
    # Create dummy inputs
    LOGGER.info("Preparing dummy inputs...")
    batch_size = 1
    dummy_image = torch.randn(batch_size, 3, INTERNAL_SIZE[0], INTERNAL_SIZE[1])
    dummy_disparity_factor = torch.tensor([0.5])  # Typical value: f_px / width
    
    # Export to ONNX
    LOGGER.info(f"Exporting to ONNX (opset {opset_version})...")
    
    input_names = ["image", "disparity_factor"]
    output_names = ["mean_vectors", "quaternions", "singular_values", "opacities", "colors"]
    
    dynamic_axes = {
        "image": {0: "batch_size"},
        "disparity_factor": {0: "batch_size"},
        "mean_vectors": {0: "batch_size"},
        "quaternions": {0: "batch_size"},
        "singular_values": {0: "batch_size"},
        "opacities": {0: "batch_size"},
        "colors": {0: "batch_size"},
    }
    
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_disparity_factor),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    
    LOGGER.info(f"ONNX model saved to {output_path}")
    
    # Verify the exported model
    LOGGER.info("Verifying ONNX model...")
    import onnx
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    LOGGER.info("ONNX model verification passed!")
    
    # Optionally simplify
    if simplify:
        try:
            import onnxsim
            LOGGER.info("Simplifying ONNX model...")
            model_simplified, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_simplified, str(output_path))
                LOGGER.info("Model simplified successfully!")
            else:
                LOGGER.warning("Simplification check failed, keeping original model")
        except ImportError:
            LOGGER.warning("onnxsim not installed, skipping simplification")
    
    # Print model info
    LOGGER.info("\n=== Model Info ===")
    LOGGER.info(f"Input shape: [B, 3, {INTERNAL_SIZE[0]}, {INTERNAL_SIZE[1]}]")
    LOGGER.info("Outputs:")
    LOGGER.info("  - mean_vectors: [B, N, 3] - 3D positions")
    LOGGER.info("  - quaternions: [B, N, 4] - rotation quaternions")
    LOGGER.info("  - singular_values: [B, N, 3] - scales")
    LOGGER.info("  - opacities: [B, N] - opacity values")
    LOGGER.info("  - colors: [B, N, 3] - RGB colors")


def verify_onnx_runtime(onnx_path: Path) -> None:
    """Verify the ONNX model works with ONNX Runtime."""
    import onnxruntime as ort
    import numpy as np
    
    LOGGER.info("\n=== Testing with ONNX Runtime ===")
    
    # Create session
    session = ort.InferenceSession(str(onnx_path))
    
    # Create test inputs
    test_image = np.random.randn(1, 3, INTERNAL_SIZE[0], INTERNAL_SIZE[1]).astype(np.float32)
    test_disparity = np.array([0.5], dtype=np.float32)
    
    # Run inference
    outputs = session.run(
        None,
        {"image": test_image, "disparity_factor": test_disparity}
    )
    
    LOGGER.info("Inference successful!")
    LOGGER.info(f"mean_vectors shape: {outputs[0].shape}")
    LOGGER.info(f"quaternions shape: {outputs[1].shape}")
    LOGGER.info(f"singular_values shape: {outputs[2].shape}")
    LOGGER.info(f"opacities shape: {outputs[3].shape}")
    LOGGER.info(f"colors shape: {outputs[4].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export ml-sharp model to ONNX format"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        default=None,
        help="Path to .pt checkpoint. If not provided, downloads default model."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("sharp_model.onnx"),
        help="Output path for ONNX model (default: sharp_model.onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)"
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX model simplification"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported model with ONNX Runtime"
    )
    
    args = parser.parse_args()
    
    # Export model
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )
    
    # Optionally verify
    if args.verify:
        verify_onnx_runtime(args.output)


if __name__ == "__main__":
    main()
