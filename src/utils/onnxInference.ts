/**
 * ONNX Runtime Web inference for ml-sharp model
 * 
 * This module provides inference capabilities using the ONNX-exported
 * ml-sharp model for generating 3D Gaussian Splatting from images.
 */

import * as ort from 'onnxruntime-web';

// Internal model resolution (from ml-sharp)
const INTERNAL_SIZE = 1536;

// Default focal length assumption (can be overridden)
const DEFAULT_FOCAL_LENGTH = 1000;

/**
 * Gaussian output from the model
 */
export interface GaussianOutput {
  /** Position of each Gaussian [N, 3] */
  meanVectors: Float32Array;
  /** Rotation quaternion [N, 4] */
  quaternions: Float32Array;
  /** Scale of each Gaussian [N, 3] */
  scales: Float32Array;
  /** Opacity values [N] */
  opacities: Float32Array;
  /** RGB colors [N, 3] */
  colors: Float32Array;
  /** Number of Gaussians */
  numGaussians: number;
}

/**
 * Configuration for the inference session
 */
export interface InferenceConfig {
  /** Path or URL to the ONNX model */
  modelPath: string;
  /** Execution provider: 'webgl', 'wasm', or 'webgpu' */
  executionProvider?: 'webgl' | 'wasm' | 'webgpu';
  /** Enable profiling for debugging */
  enableProfiling?: boolean;
}

/**
 * ML-Sharp ONNX Inference Engine
 * 
 * Handles loading the ONNX model and running inference on images
 * to generate 3D Gaussian Splatting representations.
 */
export class SharpInference {
  private session: ort.InferenceSession | null = null;
  private config: InferenceConfig;
  private isInitialized = false;

  constructor(config: InferenceConfig) {
    this.config = {
      executionProvider: 'webgl',
      enableProfiling: false,
      ...config,
    };
  }

  /**
   * Initialize the inference session
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('Initializing ONNX Runtime Web...');
    
    // Configure execution providers based on config
    const executionProviders: ort.InferenceSession.ExecutionProviderConfig[] = [];
    
    switch (this.config.executionProvider) {
      case 'webgpu':
        executionProviders.push('webgpu');
        break;
      case 'webgl':
        executionProviders.push('webgl');
        break;
      case 'wasm':
      default:
        executionProviders.push('wasm');
        break;
    }
    
    // Fallback to wasm if preferred provider fails
    if (!executionProviders.includes('wasm')) {
      executionProviders.push('wasm');
    }

    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders,
      graphOptimizationLevel: 'all',
      enableProfiling: this.config.enableProfiling,
    };

    // Probe for an external-data sidecar (.onnx.data) that accompanies
    // models exported with save_as_external_data=True.
    const dataUrl = this.config.modelPath + '.data';
    try {
      const resp = await fetch(dataUrl, { method: 'HEAD' });
      if (resp.ok) {
        sessionOptions.externalData = [dataUrl];
        console.log('External model data detected:', dataUrl);
      }
    } catch {
      // No external data file — single-file model
    }

    try {
      console.log(`Loading model from: ${this.config.modelPath}`);
      this.session = await ort.InferenceSession.create(
        this.config.modelPath,
        sessionOptions
      );
      this.isInitialized = true;
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * Run inference on an image
   * 
   * @param imageData - ImageData from canvas or image element
   * @param focalLength - Focal length in pixels (optional)
   * @returns Gaussian output with positions, rotations, scales, etc.
   */
  async infer(
    imageData: ImageData,
    focalLength?: number
  ): Promise<GaussianOutput> {
    if (!this.session) {
      throw new Error('Session not initialized. Call initialize() first.');
    }

    const { width, height, data } = imageData;
    const f_px = focalLength ?? DEFAULT_FOCAL_LENGTH;
    
    // Preprocess image: resize to INTERNAL_SIZE x INTERNAL_SIZE and normalize
    const processedImage = await this.preprocessImage(data, width, height);
    
    // Calculate disparity factor (f_px / width)
    const disparityFactor = f_px / width;

    // Create input tensors
    const imageTensor = new ort.Tensor(
      'float32',
      processedImage,
      [1, 3, INTERNAL_SIZE, INTERNAL_SIZE]
    );
    
    const disparityTensor = new ort.Tensor(
      'float32',
      new Float32Array([disparityFactor]),
      [1]
    );

    console.log('Running inference...');
    const startTime = performance.now();

    // Run inference
    const results = await this.session.run({
      image: imageTensor,
      disparity_factor: disparityTensor,
    });

    const endTime = performance.now();
    console.log(`Inference completed in ${(endTime - startTime).toFixed(2)}ms`);

    // Extract outputs
    const meanVectors = results['mean_vectors'].data as Float32Array;
    const quaternions = results['quaternions'].data as Float32Array;
    const scales = results['singular_values'].data as Float32Array;
    const opacities = results['opacities'].data as Float32Array;
    const colors = results['colors'].data as Float32Array;

    // Calculate number of Gaussians (assuming [B, N, 3] for mean_vectors)
    const numGaussians = meanVectors.length / 3;

    return {
      meanVectors,
      quaternions,
      scales,
      opacities,
      colors,
      numGaussians,
    };
  }

  /**
   * Preprocess image for model input
   * Resize to INTERNAL_SIZE x INTERNAL_SIZE and normalize to [0, 1]
   */
  private async preprocessImage(
    data: Uint8ClampedArray,
    width: number,
    height: number
  ): Promise<Float32Array> {
    // Create a canvas for resizing
    const canvas = document.createElement('canvas');
    canvas.width = INTERNAL_SIZE;
    canvas.height = INTERNAL_SIZE;
    const ctx = canvas.getContext('2d')!;

    // Create ImageData from input by copying to a new Uint8ClampedArray
    const sourceCanvas = document.createElement('canvas');
    sourceCanvas.width = width;
    sourceCanvas.height = height;
    const sourceCtx = sourceCanvas.getContext('2d')!;
    const sourceImageData = new ImageData(new Uint8ClampedArray(data), width, height);
    sourceCtx.putImageData(sourceImageData, 0, 0);

    // Draw resized image (bilinear interpolation)
    ctx.drawImage(sourceCanvas, 0, 0, INTERNAL_SIZE, INTERNAL_SIZE);
    
    // Get resized image data
    const resizedData = ctx.getImageData(0, 0, INTERNAL_SIZE, INTERNAL_SIZE).data;

    // Convert to CHW format and normalize to [0, 1]
    // Input: RGBA interleaved [H, W, 4]
    // Output: RGB planar [3, H, W]
    const numPixels = INTERNAL_SIZE * INTERNAL_SIZE;
    const result = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
      const srcIdx = i * 4;
      // R channel
      result[i] = resizedData[srcIdx] / 255.0;
      // G channel
      result[numPixels + i] = resizedData[srcIdx + 1] / 255.0;
      // B channel
      result[2 * numPixels + i] = resizedData[srcIdx + 2] / 255.0;
    }

    return result;
  }

  /**
   * Convert Gaussian output to PLY format data
   */
  gaussiansToPLY(output: GaussianOutput): ArrayBuffer {
    const { meanVectors, quaternions, scales, opacities, colors, numGaussians } = output;
    
    // PLY header
    const header = `ply
format binary_little_endian 1.0
element vertex ${numGaussians}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
`;

    const headerBytes = new TextEncoder().encode(header);
    
    // Each Gaussian: 14 floats (x, y, z, scale*3, rot*4, opacity, color*3)
    const floatsPerGaussian = 14;
    const dataSize = numGaussians * floatsPerGaussian * 4;
    
    const buffer = new ArrayBuffer(headerBytes.length + dataSize);
    const headerView = new Uint8Array(buffer, 0, headerBytes.length);
    headerView.set(headerBytes);
    
    const dataView = new DataView(buffer, headerBytes.length);
    
    for (let i = 0; i < numGaussians; i++) {
      const offset = i * floatsPerGaussian * 4;
      
      // Position
      dataView.setFloat32(offset + 0, meanVectors[i * 3 + 0], true);
      dataView.setFloat32(offset + 4, meanVectors[i * 3 + 1], true);
      dataView.setFloat32(offset + 8, meanVectors[i * 3 + 2], true);
      
      // Scale (log)
      dataView.setFloat32(offset + 12, Math.log(scales[i * 3 + 0] + 1e-6), true);
      dataView.setFloat32(offset + 16, Math.log(scales[i * 3 + 1] + 1e-6), true);
      dataView.setFloat32(offset + 20, Math.log(scales[i * 3 + 2] + 1e-6), true);
      
      // Rotation (w, x, y, z)
      dataView.setFloat32(offset + 24, quaternions[i * 4 + 3], true); // w
      dataView.setFloat32(offset + 28, quaternions[i * 4 + 0], true); // x
      dataView.setFloat32(offset + 32, quaternions[i * 4 + 1], true); // y
      dataView.setFloat32(offset + 36, quaternions[i * 4 + 2], true); // z
      
      // Opacity (logit)
      const opacity = Math.max(0.001, Math.min(0.999, opacities[i]));
      dataView.setFloat32(offset + 40, Math.log(opacity / (1 - opacity)), true);
      
      // Color (SH DC coefficients)
      const C0 = 0.28209479177387814;
      dataView.setFloat32(offset + 44, (colors[i * 3 + 0] - 0.5) / C0, true);
      dataView.setFloat32(offset + 48, (colors[i * 3 + 1] - 0.5) / C0, true);
      dataView.setFloat32(offset + 52, (colors[i * 3 + 2] - 0.5) / C0, true);
    }
    
    return buffer;
  }

  /**
   * Dispose of the session and free resources
   */
  dispose(): void {
    if (this.session) {
      this.session = null;
      this.isInitialized = false;
    }
  }
}

/**
 * Helper function to load an image and get ImageData
 */
export async function loadImageData(source: File | string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      resolve(ctx.getImageData(0, 0, img.width, img.height));
    };
    
    img.onerror = (e) => reject(new Error(`Failed to load image: ${e}`));
    
    if (typeof source === 'string') {
      img.src = source;
    } else {
      img.src = URL.createObjectURL(source);
    }
  });
}
