import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Set base path for GitHub Pages deployment
  base: '/mlsharp-web/',
  // Prevent Vite from pre-bundling onnxruntime-web so the WASM sidecar files
  // are not inlined or mangled – they must remain as separate URLs.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      // Enable SharedArrayBuffer for the multithreaded WASM backend during
      // local development.  These headers have no effect on GitHub Pages
      // (which does not support custom HTTP headers); ORT will silently fall
      // back to single-threaded WASM mode on the live site.
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
