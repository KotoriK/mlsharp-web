# SHARP Web

A web-based 3D Gaussian Splatting viewer inspired by [Apple ML-SHARP](https://github.com/apple/ml-sharp).

![SHARP Web](https://img.shields.io/badge/Built%20with-React%20%2B%20TypeScript-blue)
![WebGL](https://img.shields.io/badge/Rendering-WebGL-green)

## Overview

This project brings 3D Gaussian Splatting visualization to the web browser. It allows you to:

- 📁 Load PLY files containing 3D Gaussian data
- 🖱️ Interact with scenes using orbit camera controls
- 💾 Export modified scenes back to PLY format
- 🚀 Enjoy WebGL-accelerated rendering

## About ML-SHARP

[ML-SHARP](https://github.com/apple/ml-sharp) by Apple is a PyTorch-based project for "Sharp Monocular View Synthesis" - predicting 3D Gaussian Splatting representations from a single image in less than a second.

### Original ML-SHARP Stack

The original project uses:
- **PyTorch** - Deep learning framework
- **timm** - PyTorch Image Models for encoder backbone (DPT/ViT)
- **gsplat** - Gaussian Splatting library for CUDA rendering
- **numpy/scipy** - Numerical computations
- **plyfile** - PLY file I/O

### Web Migration Strategy

To run Gaussian Splatting in the browser, this project implements:

1. **PLY Parser** - Custom TypeScript implementation for parsing binary PLY files
2. **WebGL Renderer** - Simplified Gaussian Splatting renderer using WebGL 2.0
3. **React UI** - Modern React + TypeScript application with Vite

For full neural network inference in the browser, the model would need to be:
1. Exported from PyTorch to ONNX format
2. Run using ONNX Runtime Web (included in dependencies)

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### Usage

1. Open the application in your browser
2. Upload a PLY file from ML-SHARP or other 3DGS tools
3. Use mouse to orbit around the scene
4. Scroll to zoom in/out

## Project Structure

```
src/
├── components/           # React components
│   ├── ImageUpload.tsx   # File upload component
│   ├── GaussianViewer.tsx # WebGL viewer component
│   └── ProcessingStatus.tsx # Status display
├── utils/
│   ├── plyParser.ts      # PLY file parser
│   └── gaussianRenderer.ts # WebGL renderer
├── types/
│   └── gaussian.ts       # TypeScript type definitions
├── App.tsx               # Main application
└── main.tsx              # Entry point
```

## Technical Details

### PLY Format Support

The parser supports binary PLY files with the standard 3DGS properties:
- Position (x, y, z)
- Scale (scale_0, scale_1, scale_2) - stored as log values
- Rotation (rot_0, rot_1, rot_2, rot_3) - quaternion
- Opacity - stored as logit
- Spherical Harmonics (f_dc_0, f_dc_1, f_dc_2, f_rest_*)

### WebGL Rendering

The renderer implements:
- Point-based Gaussian rendering
- Gaussian falloff in fragment shader
- Alpha blending for transparency
- Orbit camera controls

### Limitations

- Simplified rendering compared to full 3DGS (no sorting, basic 2D projection)
- Image-to-Gaussian inference requires ONNX model (future work)
- WebGL 2.0 required (no WebGPU implementation yet)

## Future Improvements

- [ ] ONNX Runtime integration for neural network inference
- [ ] WebGPU renderer for better performance
- [ ] Proper depth sorting for Gaussians
- [ ] Higher-order spherical harmonics support
- [ ] Video export functionality

## References

- [ML-SHARP Paper](https://arxiv.org/abs/2512.10685) - Sharp Monocular View Synthesis in Less Than a Second
- [Project Page](https://apple.github.io/ml-sharp/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original 3DGS paper

## License

This project is open source. For the ML-SHARP model and code, please refer to the [original repository licenses](https://github.com/apple/ml-sharp).
