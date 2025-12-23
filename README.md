# SHARP Web

A web-based implementation of [Apple ML-SHARP](https://github.com/apple/ml-sharp) for generating 3D Gaussian Splatting from single images, running entirely in your browser.

![SHARP Web](https://img.shields.io/badge/Built%20with-React%20%2B%20TypeScript-blue)
![Viewer](https://img.shields.io/badge/Viewer-GaussianSplats3D-green)
![Inference](https://img.shields.io/badge/Inference-ONNX%20Runtime%20Web-orange)

## Overview

This project brings Apple's ML-SHARP (Sharp Monocular View Synthesis) to the web browser:

- 🖼️ **Image to 3DGS**: Upload an image and generate 3D Gaussian Splatting (requires ONNX model)
- 📁 **PLY/Splat Viewer**: Load existing Gaussian Splatting files
- 🎮 **Interactive 3D**: Full-featured viewer with [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)
- 💾 **Export**: Download results as PLY files
- 🚀 **GPU-accelerated**: Uses ONNX Runtime Web with WebGL backend

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.9+ (for model export)
- [ml-sharp](https://github.com/apple/ml-sharp) repository (for model export)

### Installation

```bash
# Install web dependencies
npm install

# Start development server
npm run dev
```

### Exporting the ONNX Model

To enable image-to-3DGS inference, you need to export the ml-sharp model:

```bash
# Clone ml-sharp
git clone https://github.com/apple/ml-sharp
cd ml-sharp

# Install ml-sharp and export dependencies
pip install -e .
pip install onnx onnxruntime onnxsim

# Export model (downloads checkpoint automatically)
python ../scripts/export_to_onnx.py -o ../public/models/sharp_model.onnx --verify

# Return to this project
cd ..
npm run dev
```

The model will be downloaded automatically from Apple's servers on first export.

## Project Structure

```
├── scripts/
│   └── export_to_onnx.py    # PyTorch to ONNX export script
├── src/
│   ├── components/
│   │   ├── GaussianViewer.tsx   # 3D viewer (GaussianSplats3D)
│   │   ├── ImageUpload.tsx      # File upload component
│   │   └── ProcessingStatus.tsx # Status display
│   ├── utils/
│   │   ├── onnxInference.ts     # ONNX Runtime inference
│   │   └── plyParser.ts         # PLY file parsing
│   └── App.tsx                  # Main application
├── public/
│   └── models/                  # Place ONNX model here
└── package.json
```

## Technical Details

### ML-SHARP Architecture

The original ml-sharp model consists of:
- **Monodepth Encoder**: DPT-based depth estimation (ViT backbone)
- **Feature Decoder**: Multi-scale feature extraction
- **Gaussian Composer**: Converts features to 3D Gaussian parameters

### Web Implementation

- **Inference**: ONNX Runtime Web with WebGL backend
- **Rendering**: [@mkkellogg/gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D)
- **UI**: React + TypeScript + Vite

### Model I/O

**Input:**
- Image: [1, 3, 1536, 1536] (resized internally)
- Disparity factor: [1] (focal_length / image_width)

**Output:**
- mean_vectors: [B, N, 3] - 3D positions
- quaternions: [B, N, 4] - rotation quaternions
- singular_values: [B, N, 3] - scales
- opacities: [B, N] - opacity values
- colors: [B, N, 3] - RGB colors

## Usage

1. **View existing PLY files**: Upload a `.ply` or `.splat` file directly
2. **Generate from image** (requires ONNX model):
   - Upload an image (JPG, PNG, WebP)
   - Wait for inference to complete
   - Explore the generated 3D Gaussians
3. **Export**: Download the result as a PLY file

## Controls

- **Left-click + drag**: Rotate camera
- **Right-click + drag**: Pan camera
- **Scroll**: Zoom in/out

## Limitations

- Model size (~500MB) may be large for some browsers
- Inference time depends on GPU capabilities
- Best results with images similar to ml-sharp training data

## References

- [ML-SHARP Paper](https://arxiv.org/abs/2512.10685) - Sharp Monocular View Synthesis in Less Than a Second
- [ML-SHARP Repository](https://github.com/apple/ml-sharp)
- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) - Three.js Gaussian Splatting viewer
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original paper

## License

This project is open source. For the ML-SHARP model and code, please refer to the [original repository licenses](https://github.com/apple/ml-sharp).
