/**
 * SHARP Web - 3D Gaussian Splatting from Images
 * 
 * A web-based application for running ml-sharp model inference
 * and viewing 3D Gaussian Splatting results.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { ImageUpload, GaussianViewer, ProcessingStatusDisplay } from './components';
import { SharpInference, loadImageData } from './utils/onnxInference';
import type { ProcessingStatus } from './types';
import './App.css';

// Model is distributed via GitHub Releases to support large file sizes
// (GitHub Pages has a 100 MB per-file limit; the ONNX weights are ~2.4 GB).
// VITE_MODEL_URL can be overridden in .env.local for local development.
const DEFAULT_MODEL_PATH =
  import.meta.env.VITE_MODEL_URL ||
  'https://github.com/KotoriK/mlsharp-web/releases/latest/download/sharp_model.onnx';

function App() {
  const [splatUrl, setSplatUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [error, setError] = useState<Error | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [modelAvailable, setModelAvailable] = useState<boolean | null>(null);
  
  const inferenceRef = useRef<SharpInference | null>(null);

  // Check if model is available on mount
  useEffect(() => {
    const checkModel = async () => {
      try {
        const response = await fetch(DEFAULT_MODEL_PATH, { method: 'HEAD' });
        setModelAvailable(response.ok);
      } catch {
        setModelAvailable(false);
      }
    };
    checkModel();
  }, []);

  const handleFileSelect = useCallback(async (file: File) => {
    setError(null);
    
    try {
      // Check file type
      const isPLY = file.name.toLowerCase().endsWith('.ply');
      const isSplat = file.name.toLowerCase().endsWith('.splat');
      const isImage = /\.(jpg|jpeg|png|webp|gif)$/i.test(file.name);

      if (isPLY || isSplat) {
        // Direct PLY/splat file - just load it
        setStatus('loading');
        setMessage('Loading Gaussian splat file...');
        setProgress(50);
        
        // Create object URL for the viewer
        const url = URL.createObjectURL(file);
        setSplatUrl(url);
        setUploadedImage(null);
        
        setStatus('complete');
        setMessage('Gaussian splat loaded');
        setProgress(100);
      } else if (isImage) {
        // Image file - run inference if model available
        setUploadedImage(URL.createObjectURL(file));
        setSplatUrl(null);
        
        if (!modelAvailable) {
          setStatus('complete');
          setMessage('Image loaded. ONNX model not available - see instructions below to enable inference.');
          setProgress(100);
          return;
        }
        
        // Initialize inference if needed
        if (!inferenceRef.current) {
          setStatus('loading');
          setMessage('Initializing ONNX Runtime...');
          setProgress(10);
          
          inferenceRef.current = new SharpInference({
            modelPath: DEFAULT_MODEL_PATH,
            executionProvider: 'webgl',
          });
          await inferenceRef.current.initialize();
        }
        
        // Load image data
        setStatus('processing');
        setMessage('Processing image...');
        setProgress(30);
        
        const imageData = await loadImageData(file);
        
        // Run inference
        setMessage('Running neural network inference...');
        setProgress(50);
        
        const output = await inferenceRef.current.infer(imageData);
        
        // Convert to PLY
        setMessage('Converting to Gaussian splats...');
        setProgress(80);
        
        const plyBuffer = inferenceRef.current.gaussiansToPLY(output);
        const blob = new Blob([plyBuffer], { type: 'application/octet-stream' });
        const url = URL.createObjectURL(blob);
        
        setSplatUrl(url);
        setStatus('complete');
        setMessage(`Generated ${output.numGaussians.toLocaleString()} Gaussians`);
        setProgress(100);
      } else {
        throw new Error('Unsupported file format. Please upload a PLY, splat, or image file.');
      }
    } catch (err) {
      console.error('Error processing file:', err);
      setStatus('error');
      setError(err instanceof Error ? err : new Error('Unknown error'));
      setMessage('Failed to process file');
    }
  }, [modelAvailable]);

  const handleReset = useCallback(() => {
    if (splatUrl) URL.revokeObjectURL(splatUrl);
    if (uploadedImage) URL.revokeObjectURL(uploadedImage);
    
    setSplatUrl(null);
    setUploadedImage(null);
    setStatus('idle');
    setProgress(0);
    setMessage('');
    setError(null);
  }, [splatUrl, uploadedImage]);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SHARP Web</h1>
        <p className="subtitle">
          3D Gaussian Splatting from Images
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
            accept="image/*,.ply,.splat"
          />
          <ProcessingStatusDisplay
            status={status}
            progress={progress}
            message={message}
            error={error}
          />
          
          {uploadedImage && (
            <div className="image-preview">
              <h4>Input Image</h4>
              <img src={uploadedImage} alt="Input" style={{ maxWidth: '100%', borderRadius: '8px' }} />
            </div>
          )}
        </section>

        <section className="viewer-section">
          <GaussianViewer 
            splatUrl={splatUrl} 
            width={800} 
            height={600}
            onLoadStart={() => {
              setStatus('rendering');
              setMessage('Loading Gaussians into viewer...');
            }}
            onLoadComplete={(count) => {
              setStatus('complete');
              setMessage(`Viewing ${count.toLocaleString()} Gaussians`);
            }}
            onLoadError={(err) => {
              setStatus('error');
              setError(err);
            }}
          />
          
          {splatUrl && (
            <div className="viewer-controls">
              <a 
                href={splatUrl} 
                download="gaussians.ply" 
                className="btn btn-primary"
              >
                Download PLY
              </a>
              <button onClick={handleReset} className="btn btn-secondary">
                Reset
              </button>
            </div>
          )}
        </section>

        <section className="info-section">
          <h2>About</h2>
          <p>
            This web application implements{' '}
            <a href="https://github.com/apple/ml-sharp" target="_blank" rel="noopener noreferrer">
              Apple's ML-SHARP
            </a>{' '}
            for generating 3D Gaussian Splatting representations from single images, 
            running entirely in your browser.
          </p>
          
          <h3>Features</h3>
          <ul>
            <li>🖼️ Upload images for 3D Gaussian generation (requires ONNX model)</li>
            <li>📁 Load existing PLY/splat files directly</li>
            <li>🎮 Interactive 3D viewing with{' '}
              <a href="https://github.com/mkkellogg/GaussianSplats3D" target="_blank" rel="noopener noreferrer">
                GaussianSplats3D
              </a>
            </li>
            <li>💾 Export results as PLY files</li>
            <li>🚀 GPU-accelerated inference with ONNX Runtime Web</li>
          </ul>

          <h3>Controls</h3>
          <ul>
            <li><strong>Left-click + drag:</strong> Rotate camera</li>
            <li><strong>Right-click + drag:</strong> Pan camera</li>
            <li><strong>Scroll:</strong> Zoom in/out</li>
          </ul>

          {modelAvailable === false && (
            <div className="model-notice">
              <h3>⚠️ ONNX Model Required for Image Inference</h3>
              <p>
                To enable image-to-3DGS inference, you need to export the ml-sharp model to ONNX format:
              </p>
              <ol>
                <li>Clone the <a href="https://github.com/apple/ml-sharp" target="_blank" rel="noopener noreferrer">ml-sharp repository</a></li>
                <li>Install dependencies: <code>pip install -e . onnx onnxruntime</code></li>
                <li>Run the export script: <code>python scripts/export_to_onnx.py -o public/models/sharp_model.onnx</code></li>
                <li>Restart the application</li>
              </ol>
              <p>You can still load existing PLY/splat files without the model.</p>
            </div>
          )}

          <h3>Technical Details</h3>
          <p>
            Built with React, TypeScript, ONNX Runtime Web for inference, and{' '}
            <a href="https://github.com/mkkellogg/GaussianSplats3D" target="_blank" rel="noopener noreferrer">
              @mkkellogg/gaussian-splats-3d
            </a>{' '}
            for rendering. The ml-sharp model can be exported to ONNX format using 
            the included Python script.
          </p>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Open source project implementing{' '}
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
