/**
 * SHARP Web - 3D Gaussian Splatting Viewer
 * 
 * A web-based viewer for 3D Gaussian Splatting files,
 * inspired by Apple's ml-sharp project.
 */

import { useState, useCallback } from 'react';
import { ImageUpload, GaussianViewer, ProcessingStatusDisplay } from './components';
import { parsePLYFile, exportToPLY } from './utils';
import type { GaussianScene, ProcessingStatus } from './types';
import './App.css';

function App() {
  const [scene, setScene] = useState<GaussianScene | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState<Error | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  const handleFileSelect = useCallback(async (file: File) => {
    setUploadedFileName(file.name);
    setError(null);
    
    try {
      // Check file type
      const isPLY = file.name.toLowerCase().endsWith('.ply');
      const isImage = /\.(jpg|jpeg|png|webp|gif)$/i.test(file.name);

      if (isPLY) {
        // Parse PLY file
        setStatus('loading');
        setMessage('Loading PLY file...');
        setProgress(10);

        setStatus('processing');
        setMessage('Parsing Gaussian data...');
        setProgress(50);

        const parsedScene = await parsePLYFile(file);
        
        setProgress(100);
        setScene(parsedScene);
        setStatus('complete');
        setMessage(`Loaded ${parsedScene.gaussians.length.toLocaleString()} Gaussians`);
      } else if (isImage) {
        // For images, show info about future ONNX support
        setStatus('processing');
        setMessage('Image processing...');
        setProgress(30);

        // Create a demo scene with simulated Gaussians from the image
        // In a full implementation, this would run ONNX inference
        await new Promise(resolve => setTimeout(resolve, 500));
        
        setStatus('complete');
        setMessage('Image loaded. Full inference requires ONNX model (coming soon).');
        setProgress(100);
        
        // Show image preview - create minimal Gaussian scene
        setScene(null);
      } else {
        throw new Error('Unsupported file format. Please upload a PLY or image file.');
      }
    } catch (err) {
      console.error('Error processing file:', err);
      setStatus('error');
      setError(err instanceof Error ? err : new Error('Unknown error'));
      setMessage('Failed to process file');
    }
  }, []);

  const handleDownload = useCallback(() => {
    if (!scene) return;
    
    const blob = exportToPLY(scene);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = uploadedFileName?.replace(/\.[^.]+$/, '_modified.ply') || 'gaussians.ply';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [scene, uploadedFileName]);

  const handleReset = useCallback(() => {
    setScene(null);
    setStatus('idle');
    setProgress(0);
    setMessage('');
    setError(null);
    setUploadedFileName(null);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SHARP Web</h1>
        <p className="subtitle">
          3D Gaussian Splatting Viewer
          <a
            href="https://github.com/apple/ml-sharp"
            target="_blank"
            rel="noopener noreferrer"
            className="source-link"
          >
            Based on Apple ML-SHARP
          </a>
        </p>
      </header>

      <main className="app-main">
        <section className="upload-section">
          <ImageUpload
            onFileSelect={handleFileSelect}
            disabled={status === 'loading' || status === 'processing'}
          />
          <ProcessingStatusDisplay
            status={status}
            progress={progress}
            message={message}
            error={error}
          />
        </section>

        <section className="viewer-section">
          <GaussianViewer scene={scene} width={800} height={600} />
          
          {scene && (
            <div className="viewer-controls">
              <button onClick={handleDownload} className="btn btn-primary">
                Download PLY
              </button>
              <button onClick={handleReset} className="btn btn-secondary">
                Reset
              </button>
            </div>
          )}
        </section>

        <section className="info-section">
          <h2>About</h2>
          <p>
            This web application allows you to view 3D Gaussian Splatting (3DGS) scenes
            directly in your browser. Upload a PLY file exported from ml-sharp or other
            3DGS tools to explore the scene in 3D.
          </p>
          
          <h3>Features</h3>
          <ul>
            <li>📁 Load PLY files with 3D Gaussian data</li>
            <li>🖱️ Interactive 3D viewing with orbit controls</li>
            <li>💾 Export modified scenes as PLY files</li>
            <li>🚀 WebGL-accelerated rendering</li>
          </ul>

          <h3>Controls</h3>
          <ul>
            <li><strong>Drag:</strong> Rotate camera</li>
            <li><strong>Scroll:</strong> Zoom in/out</li>
          </ul>

          <h3>Technical Details</h3>
          <p>
            Built with React, TypeScript, and WebGL. The renderer implements a
            simplified version of Gaussian Splatting optimized for browser performance.
          </p>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Open source project inspired by{' '}
          <a
            href="https://github.com/apple/ml-sharp"
            target="_blank"
            rel="noopener noreferrer"
          >
            Apple ML-SHARP
          </a>
          . See the{' '}
          <a
            href="https://arxiv.org/abs/2512.10685"
            target="_blank"
            rel="noopener noreferrer"
          >
            research paper
          </a>
          .
        </p>
      </footer>
    </div>
  );
}

export default App;
