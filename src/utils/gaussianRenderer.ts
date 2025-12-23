/**
 * WebGL-based 3D Gaussian Splatting Renderer
 * 
 * This implements a simplified version of Gaussian Splatting rendering
 * optimized for WebGL/WebGPU in the browser.
 */

import type { GaussianScene, CameraParams, RenderSettings } from '../types';

// Vertex shader for Gaussian splatting
// NOTE: This is a simplified implementation using gl_PointSize for rendering.
// A full implementation would use proper 2D covariance projection for elliptical Gaussians.
const VERTEX_SHADER = `#version 300 es
precision highp float;

in vec3 position;
in vec3 color;
in float opacity;
in vec3 scale;
in vec4 rotation;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 viewport;
uniform vec3 cameraPosition;

out vec4 vColor;
out vec2 vCenterOffset;
out float vOpacity;
out vec2 vConic;
out float vSize;

// Convert quaternion to rotation matrix
mat3 quaternionToMatrix(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y - w*z), 2.0*(x*z + w*y),
    2.0*(x*y + w*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - w*x),
    2.0*(x*z - w*y), 2.0*(y*z + w*x), 1.0 - 2.0*(x*x + y*y)
  );
}

void main() {
  // Transform position to view space
  vec4 viewPos = viewMatrix * vec4(position, 1.0);
  
  // Skip gaussians behind camera by moving them outside clip volume
  // We use w=0 trick to effectively discard the point
  if (viewPos.z > -0.1) {
    gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
    gl_PointSize = 0.0;
    vOpacity = 0.0;
    return;
  }
  
  // Compute 2D covariance from 3D covariance (simplified - produces circular splats)
  // A full implementation would project the 3D covariance to screen space
  mat3 rotMat = quaternionToMatrix(rotation);
  mat3 scaleMat = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
  mat3 cov3D = rotMat * scaleMat * transpose(scaleMat) * transpose(rotMat);
  
  // Project to screen space
  vec4 clipPos = projectionMatrix * viewPos;
  vec2 screenPos = clipPos.xy / clipPos.w;
  
  // Compute screen-space size
  float depth = -viewPos.z;
  float focalLength = projectionMatrix[0][0] * viewport.x * 0.5;
  float screenScale = focalLength / depth;
  
  // Simplified 2D Gaussian size (uses max scale for circular approximation)
  float maxScale = max(max(scale.x, scale.y), scale.z);
  vSize = maxScale * screenScale * 4.0;
  
  gl_Position = clipPos;
  gl_PointSize = min(vSize, 512.0); // Clamp to avoid GPU issues with large points
  
  vColor = vec4(color, 1.0);
  vOpacity = opacity;
  vCenterOffset = vec2(0.0);
  vConic = vec2(1.0 / (maxScale * maxScale * screenScale * screenScale));
}
`;

// Fragment shader for Gaussian splatting
const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColor;
in float vOpacity;
in float vSize;

out vec4 fragColor;

void main() {
  // Compute distance from center
  vec2 offset = gl_PointCoord - vec2(0.5);
  float dist2 = dot(offset, offset) * 4.0;
  
  // Gaussian falloff
  float alpha = exp(-dist2 * 2.0) * vOpacity;
  
  if (alpha < 0.01) {
    discard;
  }
  
  fragColor = vec4(vColor.rgb * alpha, alpha);
}
`;

/**
 * WebGL Gaussian Splatting Renderer
 */
export class GaussianRenderer {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram | null = null;
  private positionBuffer: WebGLBuffer | null = null;
  private colorBuffer: WebGLBuffer | null = null;
  private opacityBuffer: WebGLBuffer | null = null;
  private scaleBuffer: WebGLBuffer | null = null;
  private rotationBuffer: WebGLBuffer | null = null;
  private numGaussians: number = 0;
  private sortedIndices: Uint32Array | null = null;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.initShaders();
  }

  /**
   * Initialize shader program
   */
  private initShaders(): void {
    const gl = this.gl;

    const vertShader = this.compileShader(gl.VERTEX_SHADER, VERTEX_SHADER);
    const fragShader = this.compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SHADER);

    if (!vertShader || !fragShader) {
      throw new Error('Failed to compile shaders');
    }

    this.program = gl.createProgram();
    if (!this.program) {
      throw new Error('Failed to create program');
    }

    gl.attachShader(this.program, vertShader);
    gl.attachShader(this.program, fragShader);
    gl.linkProgram(this.program);

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(this.program);
      throw new Error(`Failed to link program: ${info}`);
    }

    gl.deleteShader(vertShader);
    gl.deleteShader(fragShader);
  }

  /**
   * Compile a shader
   */
  private compileShader(type: number, source: string): WebGLShader | null {
    const gl = this.gl;
    const shader = gl.createShader(type);
    if (!shader) return null;

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      console.error(`Shader compile error: ${info}`);
      gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  /**
   * Load a Gaussian scene into GPU buffers
   */
  loadScene(scene: GaussianScene): void {
    const { gaussians } = scene;
    this.numGaussians = gaussians.length;
    
    if (this.numGaussians === 0) return;

    // Prepare data arrays
    const positions = new Float32Array(this.numGaussians * 3);
    const colors = new Float32Array(this.numGaussians * 3);
    const opacities = new Float32Array(this.numGaussians);
    const scales = new Float32Array(this.numGaussians * 3);
    const rotations = new Float32Array(this.numGaussians * 4);

    for (let i = 0; i < this.numGaussians; i++) {
      const g = gaussians[i];
      
      // Position
      positions[i * 3 + 0] = g.position[0];
      positions[i * 3 + 1] = g.position[1];
      positions[i * 3 + 2] = g.position[2];
      
      // Color from SH DC component
      const C0 = 0.28209479177387814;
      colors[i * 3 + 0] = Math.max(0, Math.min(1, (g.sh[0] ?? 0) * C0 + 0.5));
      colors[i * 3 + 1] = Math.max(0, Math.min(1, (g.sh[1] ?? 0) * C0 + 0.5));
      colors[i * 3 + 2] = Math.max(0, Math.min(1, (g.sh[2] ?? 0) * C0 + 0.5));
      
      // Opacity
      opacities[i] = g.opacity;
      
      // Scale
      scales[i * 3 + 0] = g.scale[0];
      scales[i * 3 + 1] = g.scale[1];
      scales[i * 3 + 2] = g.scale[2];
      
      // Rotation
      rotations[i * 4 + 0] = g.rotation[0];
      rotations[i * 4 + 1] = g.rotation[1];
      rotations[i * 4 + 2] = g.rotation[2];
      rotations[i * 4 + 3] = g.rotation[3];
    }

    // Create and upload buffers
    this.positionBuffer = this.createBuffer(positions);
    this.colorBuffer = this.createBuffer(colors);
    this.opacityBuffer = this.createBuffer(opacities);
    this.scaleBuffer = this.createBuffer(scales);
    this.rotationBuffer = this.createBuffer(rotations);
    
    // Initialize sorted indices
    this.sortedIndices = new Uint32Array(this.numGaussians);
    for (let i = 0; i < this.numGaussians; i++) {
      this.sortedIndices[i] = i;
    }
  }

  /**
   * Create a WebGL buffer
   */
  private createBuffer(data: Float32Array): WebGLBuffer {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) {
      throw new Error('Failed to create buffer');
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    return buffer;
  }

  /**
   * Render the scene
   */
  render(camera: CameraParams, settings: RenderSettings): void {
    const gl = this.gl;
    
    if (!this.program || this.numGaussians === 0) return;

    // Set viewport
    gl.viewport(0, 0, settings.width, settings.height);
    
    // Clear
    gl.clearColor(
      settings.backgroundColor[0],
      settings.backgroundColor[1],
      settings.backgroundColor[2],
      1.0
    );
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    
    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    
    // Use program
    gl.useProgram(this.program);

    // Compute view matrix
    const viewMatrix = this.computeViewMatrix(camera);
    
    // Compute projection matrix
    const projectionMatrix = this.computeProjectionMatrix(
      camera.fov,
      settings.width / settings.height,
      camera.near,
      camera.far
    );

    // Set uniforms
    const viewLoc = gl.getUniformLocation(this.program, 'viewMatrix');
    const projLoc = gl.getUniformLocation(this.program, 'projectionMatrix');
    const viewportLoc = gl.getUniformLocation(this.program, 'viewport');
    const cameraPosLoc = gl.getUniformLocation(this.program, 'cameraPosition');

    gl.uniformMatrix4fv(viewLoc, false, viewMatrix);
    gl.uniformMatrix4fv(projLoc, false, projectionMatrix);
    gl.uniform2f(viewportLoc, settings.width, settings.height);
    gl.uniform3fv(cameraPosLoc, camera.position);

    // Bind attributes
    this.bindAttribute('position', this.positionBuffer!, 3);
    this.bindAttribute('color', this.colorBuffer!, 3);
    this.bindAttribute('opacity', this.opacityBuffer!, 1);
    this.bindAttribute('scale', this.scaleBuffer!, 3);
    this.bindAttribute('rotation', this.rotationBuffer!, 4);

    // Draw
    gl.drawArrays(gl.POINTS, 0, this.numGaussians);
  }

  /**
   * Bind an attribute to a buffer
   */
  private bindAttribute(name: string, buffer: WebGLBuffer, size: number): void {
    const gl = this.gl;
    const loc = gl.getAttribLocation(this.program!, name);
    if (loc === -1) return;
    
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }

  /**
   * Compute view matrix from camera parameters
   */
  private computeViewMatrix(camera: CameraParams): Float32Array {
    const [px, py, pz] = camera.position;
    const [tx, ty, tz] = camera.target;
    const [ux, uy, uz] = camera.up;

    // Forward vector
    let fx = tx - px;
    let fy = ty - py;
    let fz = tz - pz;
    const fLen = Math.sqrt(fx * fx + fy * fy + fz * fz);
    fx /= fLen; fy /= fLen; fz /= fLen;

    // Right vector
    let rx = fy * uz - fz * uy;
    let ry = fz * ux - fx * uz;
    let rz = fx * uy - fy * ux;
    const rLen = Math.sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rLen; ry /= rLen; rz /= rLen;

    // Up vector
    const nux = ry * fz - rz * fy;
    const nuy = rz * fx - rx * fz;
    const nuz = rx * fy - ry * fx;

    return new Float32Array([
      rx, nux, -fx, 0,
      ry, nuy, -fy, 0,
      rz, nuz, -fz, 0,
      -(rx * px + ry * py + rz * pz),
      -(nux * px + nuy * py + nuz * pz),
      fx * px + fy * py + fz * pz,
      1
    ]);
  }

  /**
   * Compute projection matrix
   */
  private computeProjectionMatrix(
    fov: number,
    aspect: number,
    near: number,
    far: number
  ): Float32Array {
    const f = 1.0 / Math.tan(fov * Math.PI / 360);
    const nf = 1 / (near - far);

    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0
    ]);
  }

  /**
   * Dispose of WebGL resources
   */
  dispose(): void {
    const gl = this.gl;
    
    if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
    if (this.colorBuffer) gl.deleteBuffer(this.colorBuffer);
    if (this.opacityBuffer) gl.deleteBuffer(this.opacityBuffer);
    if (this.scaleBuffer) gl.deleteBuffer(this.scaleBuffer);
    if (this.rotationBuffer) gl.deleteBuffer(this.rotationBuffer);
    if (this.program) gl.deleteProgram(this.program);
  }
}
