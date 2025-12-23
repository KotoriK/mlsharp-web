/**
 * WebGL Gaussian Splatting Viewer Component
 * 
 * Renders 3D Gaussian Splatting scenes using WebGL
 */

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import type { GaussianScene, CameraParams, RenderSettings } from '../types';
import { GaussianRenderer } from '../utils/gaussianRenderer';

interface GaussianViewerProps {
  scene: GaussianScene | null;
  width?: number;
  height?: number;
}

const DEFAULT_CAMERA: Omit<CameraParams, 'position'> = {
  target: [0, 0, 0],
  up: [0, 1, 0],
  fov: 50,
  near: 0.1,
  far: 100,
};

const DEFAULT_SETTINGS: RenderSettings = {
  width: 800,
  height: 600,
  backgroundColor: [0.1, 0.1, 0.1],
  sortGaussians: true,
  enableAntialiasing: true,
};

export function GaussianViewer({
  scene,
  width = 800,
  height = 600,
}: GaussianViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<GaussianRenderer | null>(null);
  const animationRef = useRef<number>(0);
  
  const [isDragging, setIsDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [orbitAngles, setOrbitAngles] = useState({ theta: 0, phi: Math.PI / 4 });
  const [zoom, setZoom] = useState(3);

  // Compute camera position from orbit angles
  const camera = useMemo<CameraParams>(() => {
    const { theta, phi } = orbitAngles;
    const x = zoom * Math.sin(phi) * Math.cos(theta);
    const y = zoom * Math.cos(phi);
    const z = zoom * Math.sin(phi) * Math.sin(theta);
    
    return {
      ...DEFAULT_CAMERA,
      position: [x, y, z],
    };
  }, [orbitAngles, zoom]);

  // Initialize WebGL renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
    });

    if (!gl) {
      console.error('WebGL2 not supported');
      return;
    }

    try {
      rendererRef.current = new GaussianRenderer(gl);
    } catch (error) {
      console.error('Failed to initialize renderer:', error);
    }

    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Load scene when it changes
  useEffect(() => {
    if (rendererRef.current && scene) {
      rendererRef.current.loadScene(scene);
    }
  }, [scene]);

  // Animation loop
  useEffect(() => {
    const render = () => {
      if (rendererRef.current && scene) {
        rendererRef.current.render(camera, {
          ...DEFAULT_SETTINGS,
          width,
          height,
        });
      }
      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [camera, scene, width, height]);

  // Mouse handlers for orbit controls
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  }, []);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isDragging) return;

      const dx = e.clientX - lastMouse.x;
      const dy = e.clientY - lastMouse.y;

      setOrbitAngles(prev => ({
        theta: prev.theta + dx * 0.01,
        phi: Math.max(0.1, Math.min(Math.PI - 0.1, prev.phi + dy * 0.01)),
      }));

      setLastMouse({ x: e.clientX, y: e.clientY });
    },
    [isDragging, lastMouse]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(prev => Math.max(0.5, Math.min(20, prev + e.deltaY * 0.01)));
  }, []);

  return (
    <div className="gaussian-viewer">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
        style={{
          cursor: isDragging ? 'grabbing' : 'grab',
          display: 'block',
        }}
      />
      {!scene && (
        <div className="viewer-placeholder">
          <p>Upload an image or PLY file to view</p>
        </div>
      )}
      {scene && (
        <div className="viewer-info">
          <span>{scene.gaussians.length.toLocaleString()} Gaussians</span>
        </div>
      )}
    </div>
  );
}
