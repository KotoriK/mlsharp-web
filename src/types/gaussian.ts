/**
 * Type definitions for 3D Gaussian Splatting data
 * Based on the apple/ml-sharp implementation
 */

/**
 * Single 3D Gaussian primitive
 */
export interface Gaussian3D {
  /** Position in 3D space (x, y, z) */
  position: [number, number, number];
  /** Scale (sx, sy, sz) */
  scale: [number, number, number];
  /** Rotation quaternion (x, y, z, w) */
  rotation: [number, number, number, number];
  /** Spherical harmonics coefficients for color */
  sh: number[];
  /** Opacity value (0-1) */
  opacity: number;
}

/**
 * Collection of 3D Gaussians representing a scene
 */
export interface GaussianScene {
  gaussians: Gaussian3D[];
  metadata: SceneMetadata;
}

/**
 * Scene metadata
 */
export interface SceneMetadata {
  /** Focal length in pixels */
  focalLength: number;
  /** Image dimensions [width, height] */
  imageDimensions: [number, number];
  /** Color space used */
  colorSpace: 'sRGB' | 'linearRGB';
}

/**
 * PLY file header information
 */
export interface PLYHeader {
  format: string;
  numVertices: number;
  properties: PLYProperty[];
}

/**
 * PLY property definition
 */
export interface PLYProperty {
  name: string;
  type: string;
}

/**
 * Camera parameters for rendering
 */
export interface CameraParams {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov: number;
  near: number;
  far: number;
}

/**
 * Render settings
 */
export interface RenderSettings {
  width: number;
  height: number;
  backgroundColor: [number, number, number];
  sortGaussians: boolean;
  enableAntialiasing: boolean;
}

/**
 * Processing status
 */
export type ProcessingStatus =
  | 'idle'
  | 'loading'
  | 'processing'
  | 'rendering'
  | 'complete'
  | 'error';

/**
 * Processing result
 */
export interface ProcessingResult {
  status: ProcessingStatus;
  progress: number;
  message: string;
  scene?: GaussianScene;
  error?: Error;
}
