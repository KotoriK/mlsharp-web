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
  /** Path or URL to the ONNX model, or a pre-loaded ArrayBuffer / Uint8Array */
  modelPath: string | ArrayBuffer | Uint8Array;
  /** Execution provider: 'wasm' or 'webgpu' */
  executionProvider?: 'wasm' | 'webgpu';
  /** Enable profiling for debugging */
  enableProfiling?: boolean;
  /**
   * Pre-loaded external data buffer.
   * When provided the network fetch of the .data sidecar is skipped entirely,
   * which is required when the model is loaded from a local file (no URL).
   */
  externalDataBuffer?: ArrayBuffer;
  /**
   * The path token stored inside the .onnx file that refers to the external
   * data.  Must match exactly; defaults to "sharp_model.onnx.data".
   * Only used when externalDataBuffer is provided.
   */
  externalDataFileName?: string;
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
      executionProvider: 'webgpu',
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

    // Probe for / load external-data sidecar.
    // When externalDataBuffer is provided (local file) we skip network requests.
    // When modelPath is a URL we attempt to find the .data sidecar on the server.
    if (this.config.externalDataBuffer !== undefined) {
      const path = this.config.externalDataFileName ?? 'sharp_model.onnx.data';
      sessionOptions.externalData = [{ path, data: this.config.externalDataBuffer }];
      console.log(`External model data provided locally (${this.config.externalDataBuffer.byteLength} bytes)`);
    } else if (typeof this.config.modelPath === 'string') {
      // Probe for an external-data sidecar that accompanies models exported
      // with save_as_external_data=True. Two layouts are supported:
      //   • Chunked: <model>.onnx.data.0000, .0001, … (used when the data
      //     file exceeds the 2 GB GitHub Release upload limit)
      //   • Single:  <model>.onnx.data
      const baseDataUrl = this.config.modelPath + '.data';
      const externalDataInfo = await SharpInference.fetchExternalData(baseDataUrl);
      if (externalDataInfo.type === 'url') {
        sessionOptions.externalData = [externalDataInfo.url];
        console.log('External model data detected:', externalDataInfo.url);
      } else if (externalDataInfo.type === 'buffer') {
        // The path must match the `location` field stored inside the .onnx file.
        const dataFileName = baseDataUrl.split('/').pop() ?? baseDataUrl;
        sessionOptions.externalData = [{ path: dataFileName, data: externalDataInfo.data }];
        console.log(`External model data loaded from ${externalDataInfo.parts} chunk(s), ` +
          `total ${externalDataInfo.data.byteLength} bytes`);
      }
    }
    // If modelPath is a buffer and no externalDataBuffer is provided, the model
    // is assumed to be fully self-contained (no external data file).

    try {
      if (typeof this.config.modelPath === 'string') {
        console.log(`Loading model from: ${this.config.modelPath}`);
      } else {
        console.log('Loading model from local buffer...');
      }
      // InferenceSession.create has separate overloads for string, ArrayBuffer, and Uint8Array.
      // Use explicit type-narrowing branches so TypeScript resolves the correct overload.
      if (typeof this.config.modelPath === 'string') {
        this.session = await ort.InferenceSession.create(this.config.modelPath, sessionOptions);
      } else if (this.config.modelPath instanceof Uint8Array) {
        this.session = await ort.InferenceSession.create(this.config.modelPath, sessionOptions);
      } else {
        this.session = await ort.InferenceSession.create(this.config.modelPath, sessionOptions);
      }
      this.isInitialized = true;
      console.log('Model loaded successfully');
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * Probe for ONNX external-data files and load them.
   *
   * Supports two layouts produced by the export workflow:
   *  - Chunked: `<baseDataUrl>.0000`, `.0001`, … – used when the data file
   *    exceeds the 2 GB GitHub Releases upload limit.  All chunks are
   *    downloaded and concatenated into a single ArrayBuffer.
   *  - Single:  `<baseDataUrl>` – returned as a URL so ort-web can stream it
   *    directly without buffering the whole file in JS heap.
   *
   * Returns `{ type: 'none' }` when no external data is present.
   */
  private static async fetchExternalData(baseDataUrl: string): Promise<
    | { type: 'none' }
    | { type: 'url'; url: string }
    | { type: 'buffer'; data: ArrayBuffer; parts: number }
  > {
    // 1. Check for chunked format (.data.0000, .data.0001, …)
    try {
      const chunk0Url = `${baseDataUrl}.0000`;
      const probe = await fetch(chunk0Url, { method: 'HEAD' });
      if (probe.ok) {
        console.log('Chunked external model data detected, downloading parts…');
        const chunks: ArrayBuffer[] = [];
        for (let idx = 0; ; idx++) {
          const suffix = String(idx).padStart(4, '0');
          const partUrl = `${baseDataUrl}.${suffix}`;
          const partProbe = await fetch(partUrl, { method: 'HEAD' });
          if (!partProbe.ok) break;
          console.log(`  Downloading chunk ${idx}…`);
          const partResp = await fetch(partUrl);
          chunks.push(await partResp.arrayBuffer());
        }
        // Concatenate all parts using Blob to avoid holding duplicate
        // copies in memory, preventing RangeError on large data files.
        const blob = new Blob(chunks);
        const merged = await blob.arrayBuffer();
        return { type: 'buffer', data: merged, parts: chunks.length };
      }
    } catch {
      // Chunked probe failed — fall through to single-file check.
    }

    // 2. Check for single data file.
    try {
      const probe = await fetch(baseDataUrl, { method: 'HEAD' });
      if (probe.ok) {
        return { type: 'url', url: baseDataUrl };
      }
    } catch {
      // No external data.
    }

    return { type: 'none' };
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
