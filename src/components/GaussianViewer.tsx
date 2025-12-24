/**
 * 3D Gaussian Splatting Viewer Component
 * 
 * Uses @mkkellogg/gaussian-splats-3d for high-quality Gaussian Splatting rendering
 */

import { useRef, useEffect, useState } from 'react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

interface GaussianViewerProps {
  /** URL or path to the PLY/splat file to load */
  splatUrl: string | null;
  width?: number;
  height?: number;
  /** Called when loading starts */
  onLoadStart?: () => void;
  /** Called when loading completes */
  onLoadComplete?: (gaussianCount: number) => void;
  /** Called on loading error */
  onLoadError?: (error: Error) => void;
}

export function GaussianViewer({
  splatUrl,
  width = 800,
  height = 600,
  onLoadStart,
  onLoadComplete,
  onLoadError,
}: GaussianViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<GaussianSplats3D.Viewer | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [gaussianCount, setGaussianCount] = useState(0);

  // Initialize viewer
  useEffect(() => {
    if (!containerRef.current) return;

    // Create the Gaussian Splats viewer
    const viewer = new GaussianSplats3D.Viewer({
      cameraUp: [0, 1, 0],
      initialCameraPosition: [0, 0, 5],
      initialCameraLookAt: [0, 0, 0],
      rootElement: containerRef.current,
      selfDrivenMode: true,
      useBuiltInControls: true,
      dynamicScene: false,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Gradual,
      gpuAcceleratedSort: true,
      sharedMemoryForWorkers: false,
    });

    viewerRef.current = viewer;

    // Start the render loop
    viewer.start();

    return () => {
      viewer.dispose();
      viewerRef.current = null;
    };
  }, []);

  // Update viewer size
  useEffect(() => {
    if (!containerRef.current) return;
    
    const canvas = containerRef.current.querySelector('canvas');
    if (canvas) {
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
  }, [width, height]);

  // Load splat file when URL changes
  useEffect(() => {
    if (!viewerRef.current || !splatUrl) return;

    const loadScene = async () => {
      setIsLoading(true);
      onLoadStart?.();

      try {
        // Clear existing scene
        await viewerRef.current!.removeSplatScenes();

        // Load new scene
        await viewerRef.current!.addSplatScene(splatUrl, {
          showLoadingUI: true,
          progressiveLoad: true,
        });

        // Get Gaussian count from the viewer
        const count = viewerRef.current!.getSplatCount() ?? 0;
        setGaussianCount(count);
        
        setIsLoading(false);
        onLoadComplete?.(count);
      } catch (error) {
        setIsLoading(false);
        onLoadError?.(error instanceof Error ? error : new Error('Failed to load splat'));
        console.error('Error loading splat:', error);
      }
    };

    loadScene();
  }, [splatUrl, onLoadStart, onLoadComplete, onLoadError]);

  return (
    <div className="gaussian-viewer">
      <div
        ref={containerRef}
        style={{
          width: `${width}px`,
          height: `${height}px`,
          background: '#0a0a0a',
          borderRadius: '12px',
          overflow: 'hidden',
        }}
      />
      {!splatUrl && (
        <div className="viewer-placeholder">
          <p>Upload a PLY file or run inference to view</p>
        </div>
      )}
      {isLoading && (
        <div className="viewer-loading">
          <div className="spinner" />
          <p>Loading Gaussians...</p>
        </div>
      )}
      {splatUrl && !isLoading && gaussianCount > 0 && (
        <div className="viewer-info">
          <span>{gaussianCount.toLocaleString()} Gaussians</span>
        </div>
      )}
    </div>
  );
}
