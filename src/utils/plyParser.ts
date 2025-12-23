/**
 * PLY file parser for 3D Gaussian Splatting files
 * Based on the apple/ml-sharp PLY format
 */

import type { Gaussian3D, GaussianScene, PLYHeader, PLYProperty, SceneMetadata } from '../types';

/**
 * Parse PLY header to extract format and property information
 */
function parsePLYHeader(headerText: string): PLYHeader {
  const lines = headerText.split('\n');
  let format = '';
  let numVertices = 0;
  const properties: PLYProperty[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    
    if (trimmed.startsWith('format')) {
      format = trimmed.split(' ')[1];
    } else if (trimmed.startsWith('element vertex')) {
      numVertices = parseInt(trimmed.split(' ')[2], 10);
    } else if (trimmed.startsWith('property')) {
      const parts = trimmed.split(' ');
      properties.push({
        type: parts[1],
        name: parts[2],
      });
    }
  }

  return { format, numVertices, properties };
}

/**
 * Get byte size for PLY data type
 */
function getTypeSize(type: string): number {
  switch (type) {
    case 'float':
    case 'float32':
      return 4;
    case 'double':
    case 'float64':
      return 8;
    case 'uchar':
    case 'uint8':
    case 'char':
    case 'int8':
      return 1;
    case 'ushort':
    case 'uint16':
    case 'short':
    case 'int16':
      return 2;
    case 'uint':
    case 'uint32':
    case 'int':
    case 'int32':
      return 4;
    default:
      throw new Error(`Unknown PLY type: ${type}`);
  }
}

/**
 * Read a value from DataView based on type
 */
function readValue(
  dataView: DataView,
  offset: number,
  type: string,
  littleEndian: boolean
): number {
  switch (type) {
    case 'float':
    case 'float32':
      return dataView.getFloat32(offset, littleEndian);
    case 'double':
    case 'float64':
      return dataView.getFloat64(offset, littleEndian);
    case 'uchar':
    case 'uint8':
      return dataView.getUint8(offset);
    case 'char':
    case 'int8':
      return dataView.getInt8(offset);
    case 'ushort':
    case 'uint16':
      return dataView.getUint16(offset, littleEndian);
    case 'short':
    case 'int16':
      return dataView.getInt16(offset, littleEndian);
    case 'uint':
    case 'uint32':
      return dataView.getUint32(offset, littleEndian);
    case 'int':
    case 'int32':
      return dataView.getInt32(offset, littleEndian);
    default:
      throw new Error(`Unknown PLY type: ${type}`);
  }
}

/**
 * Parse binary PLY data to extract Gaussian properties
 */
function parseBinaryPLY(
  data: ArrayBuffer,
  headerEndIndex: number,
  header: PLYHeader
): Map<string, Float32Array> {
  const littleEndian = header.format === 'binary_little_endian';
  const dataView = new DataView(data, headerEndIndex);
  
  // Calculate stride (bytes per vertex)
  const stride = header.properties.reduce((sum, prop) => sum + getTypeSize(prop.type), 0);
  
  // Create result arrays
  const result = new Map<string, Float32Array>();
  for (const prop of header.properties) {
    result.set(prop.name, new Float32Array(header.numVertices));
  }

  // Parse each vertex
  for (let i = 0; i < header.numVertices; i++) {
    let offset = i * stride;
    
    for (const prop of header.properties) {
      const value = readValue(dataView, offset, prop.type, littleEndian);
      result.get(prop.name)![i] = value;
      offset += getTypeSize(prop.type);
    }
  }

  return result;
}

/**
 * Convert spherical harmonics DC component to RGB
 * SH0 = (r, g, b) / C0 where C0 = 0.28209479177387814
 */
export function sh0ToRGB(sh0: [number, number, number]): [number, number, number] {
  const C0 = 0.28209479177387814;
  return [
    Math.max(0, Math.min(1, sh0[0] * C0 + 0.5)),
    Math.max(0, Math.min(1, sh0[1] * C0 + 0.5)),
    Math.max(0, Math.min(1, sh0[2] * C0 + 0.5)),
  ];
}

/**
 * Parse a PLY file containing 3D Gaussian Splatting data
 */
export async function parsePLYFile(file: File): Promise<GaussianScene> {
  const arrayBuffer = await file.arrayBuffer();
  return parsePLYBuffer(arrayBuffer);
}

/**
 * Parse PLY data from ArrayBuffer
 */
export function parsePLYBuffer(buffer: ArrayBuffer): GaussianScene {
  const textDecoder = new TextDecoder();
  
  // Find header end
  const maxHeaderSize = Math.min(buffer.byteLength, 4096);
  const headerBytes = new Uint8Array(buffer, 0, maxHeaderSize);
  const headerText = textDecoder.decode(headerBytes);
  
  const endHeaderIndex = headerText.indexOf('end_header');
  if (endHeaderIndex === -1) {
    throw new Error('Invalid PLY file: no end_header found');
  }
  
  // Find the actual byte position of end_header
  const headerEndIndex = new TextEncoder().encode(
    headerText.substring(0, endHeaderIndex + 'end_header'.length + 1)
  ).length;
  
  // Parse header
  const headerSection = headerText.substring(0, endHeaderIndex + 'end_header'.length);
  const header = parsePLYHeader(headerSection);
  
  if (!header.format.startsWith('binary')) {
    throw new Error('Only binary PLY format is supported');
  }

  // Parse binary data
  const propertyData = parseBinaryPLY(buffer, headerEndIndex, header);
  
  // Convert to Gaussian3D array
  const gaussians: Gaussian3D[] = [];
  
  for (let i = 0; i < header.numVertices; i++) {
    // Position
    const x = propertyData.get('x')?.[i] ?? 0;
    const y = propertyData.get('y')?.[i] ?? 0;
    const z = propertyData.get('z')?.[i] ?? 0;
    
    // Scale (log scale in PLY files)
    const scaleX = Math.exp(propertyData.get('scale_0')?.[i] ?? 0);
    const scaleY = Math.exp(propertyData.get('scale_1')?.[i] ?? 0);
    const scaleZ = Math.exp(propertyData.get('scale_2')?.[i] ?? 0);
    
    // Rotation quaternion (stored as w, x, y, z in some formats)
    const rot0 = propertyData.get('rot_0')?.[i] ?? 1;
    const rot1 = propertyData.get('rot_1')?.[i] ?? 0;
    const rot2 = propertyData.get('rot_2')?.[i] ?? 0;
    const rot3 = propertyData.get('rot_3')?.[i] ?? 0;
    
    // Normalize quaternion
    const norm = Math.sqrt(rot0*rot0 + rot1*rot1 + rot2*rot2 + rot3*rot3);
    const qw = rot0 / norm;
    const qx = rot1 / norm;
    const qy = rot2 / norm;
    const qz = rot3 / norm;
    
    // Opacity (stored as logit in some formats)
    const opacityRaw = propertyData.get('opacity')?.[i] ?? 0;
    const opacity = 1 / (1 + Math.exp(-opacityRaw)); // sigmoid
    
    // Spherical harmonics (DC component for color)
    const sh0_r = propertyData.get('f_dc_0')?.[i] ?? 0;
    const sh0_g = propertyData.get('f_dc_1')?.[i] ?? 0;
    const sh0_b = propertyData.get('f_dc_2')?.[i] ?? 0;
    
    // Collect all SH coefficients
    const sh: number[] = [sh0_r, sh0_g, sh0_b];
    
    // Add higher order SH if present
    for (let j = 0; j < 45; j++) {
      const shValue = propertyData.get(`f_rest_${j}`)?.[i];
      if (shValue !== undefined) {
        sh.push(shValue);
      }
    }
    
    gaussians.push({
      position: [x, y, z],
      scale: [scaleX, scaleY, scaleZ],
      rotation: [qx, qy, qz, qw],
      sh,
      opacity,
    });
  }

  // Default metadata
  const metadata: SceneMetadata = {
    focalLength: 500,
    imageDimensions: [1024, 1024],
    colorSpace: 'linearRGB',
  };

  return { gaussians, metadata };
}

/**
 * Export Gaussian scene to PLY format (for download)
 */
export function exportToPLY(scene: GaussianScene): Blob {
  const { gaussians } = scene;
  const numVertices = gaussians.length;
  
  // Build header
  const header = `ply
format binary_little_endian 1.0
element vertex ${numVertices}
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
  
  // Calculate data size: 14 floats per gaussian
  const floatsPerGaussian = 14;
  const dataSize = numVertices * floatsPerGaussian * 4;
  
  const buffer = new ArrayBuffer(headerBytes.length + dataSize);
  const headerView = new Uint8Array(buffer, 0, headerBytes.length);
  headerView.set(headerBytes);
  
  const dataView = new DataView(buffer, headerBytes.length);
  
  for (let i = 0; i < numVertices; i++) {
    const g = gaussians[i];
    const offset = i * floatsPerGaussian * 4;
    
    // Position
    dataView.setFloat32(offset + 0, g.position[0], true);
    dataView.setFloat32(offset + 4, g.position[1], true);
    dataView.setFloat32(offset + 8, g.position[2], true);
    
    // Scale (log)
    dataView.setFloat32(offset + 12, Math.log(g.scale[0]), true);
    dataView.setFloat32(offset + 16, Math.log(g.scale[1]), true);
    dataView.setFloat32(offset + 20, Math.log(g.scale[2]), true);
    
    // Rotation (w, x, y, z)
    dataView.setFloat32(offset + 24, g.rotation[3], true); // w
    dataView.setFloat32(offset + 28, g.rotation[0], true); // x
    dataView.setFloat32(offset + 32, g.rotation[1], true); // y
    dataView.setFloat32(offset + 36, g.rotation[2], true); // z
    
    // Opacity (logit)
    const logitOpacity = Math.log(g.opacity / (1 - Math.max(0.001, Math.min(0.999, g.opacity))));
    dataView.setFloat32(offset + 40, logitOpacity, true);
    
    // SH DC
    dataView.setFloat32(offset + 44, g.sh[0] ?? 0, true);
    dataView.setFloat32(offset + 48, g.sh[1] ?? 0, true);
    dataView.setFloat32(offset + 52, g.sh[2] ?? 0, true);
  }
  
  return new Blob([buffer], { type: 'application/octet-stream' });
}
