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


def patch_mlsharp_for_onnx() -> None:
    """
    Apply monkey-patches to ml-sharp modules for ONNX-compatible tracing.

    The ONNX tracer (TorchScript-based) cannot record Python-level operations
    as ONNX math nodes. When a tensor dimension (e.g. image width H/W) is
    extracted as a Python scalar via int()/float(), the tracer treats all
    downstream results as hard-coded constants, inflating the model size.

    These patches keep dynamic-dimension computations in the PyTorch tensor
    graph so they are exported as proper ONNX operators instead of constants.
    """
    import sharp.models.initializer as init_mod

    # --- Patch _create_base_xy ---
    # Original uses `depth.shape` values directly as Python ints in arithmetic,
    # causing coordinate grids to be baked as large constant tensors.
    # This version wraps width/height in scalar tensors so divisions become
    # ONNX Div nodes instead of folded constants.
    def _create_base_xy_patched(
        depth: torch.Tensor, stride: int, num_layers: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = depth.device
        dtype = depth.dtype
        batch_size = depth.shape[0]
        image_height = depth.shape[2]
        image_width = depth.shape[3]

        # Keep width/height as scalar tensors so that the division is recorded
        # as an ONNX Div operation rather than being folded into a constant.
        width_t = torch.tensor(image_width, dtype=dtype, device=device)
        height_t = torch.tensor(image_height, dtype=dtype, device=device)

        xx = torch.arange(0.5 * stride, image_width, stride, device=device, dtype=dtype)
        xx = 2.0 * xx / width_t - 1.0

        yy = torch.arange(0.5 * stride, image_height, stride, device=device, dtype=dtype)
        yy = 2.0 * yy / height_t - 1.0

        xx, yy = torch.meshgrid(xx, yy, indexing="xy")
        base_x_ndc = xx[None, None, None].expand(batch_size, 1, num_layers, -1, -1)
        base_y_ndc = yy[None, None, None].expand(batch_size, 1, num_layers, -1, -1)

        return base_x_ndc, base_y_ndc

    init_mod._create_base_xy = _create_base_xy_patched
    LOGGER.info("Patched initializer._create_base_xy for ONNX tracing")

    # --- Patch _create_base_scale ---
    # Ensure disparity_scale_factor stays in the tensor graph even if the
    # caller computed it as a Python float.
    _orig_create_base_scale = init_mod._create_base_scale

    def _create_base_scale_patched(
        disparity: torch.Tensor, disparity_scale_factor: float,
    ) -> torch.Tensor:
        # Wrap the scalar in a tensor so the multiply is an ONNX Mul node.
        if not isinstance(disparity_scale_factor, torch.Tensor):
            disparity_scale_factor = torch.tensor(
                disparity_scale_factor, dtype=disparity.dtype, device=disparity.device,
            )
        inverse_disparity = torch.ones_like(disparity) / disparity
        return inverse_disparity * disparity_scale_factor

    init_mod._create_base_scale = _create_base_scale_patched
    LOGGER.info("Patched initializer._create_base_scale for ONNX tracing")


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
    
    # Apply monkey-patches for ONNX-compatible tracing before model creation
    patch_mlsharp_for_onnx()
    
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
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    
    LOGGER.info(f"ONNX model saved to {output_path}")
    
    # Convert to external-data format so that the .onnx protobuf stays
    # well under the 2 GB protobuf serialisation limit.  Tensor
    # initializers are moved into a companion .data sidecar file; the
    # .onnx file retains only the lightweight graph structure.
    import onnx
    
    data_filename = output_path.name + ".data"
    LOGGER.info("Converting model to external data format...")
    model = onnx.load(str(output_path))
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
    )
    del model  # free memory; save_model mutated the proto in-place
    LOGGER.info(f"External data written to {output_path.parent / data_filename}")
    
    # Verify the exported model (file-path API resolves external data
    # from the same directory automatically).
    LOGGER.info("Verifying ONNX model...")
    onnx.checker.check_model(str(output_path))
    LOGGER.info("ONNX model verification passed!")
    
    # Optionally simplify
    if simplify:
        try:
            import onnxsim
        except ImportError:
            LOGGER.warning("onnxsim not installed, skipping simplification")
        else:
            try:
                LOGGER.info("Simplifying ONNX model...")
                model_simplified, check = onnxsim.simplify(str(output_path))
                if check:
                    onnx.save_model(
                        model_simplified,
                        str(output_path),
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=data_filename,
                        size_threshold=1024,
                    )
                    LOGGER.info("Model simplified successfully!")
                else:
                    LOGGER.warning("Simplification check failed, keeping original model")
            except Exception as exc:
                LOGGER.warning("Simplification failed: %s", exc)
                LOGGER.info("Keeping original (unsimplified) model")
    
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
