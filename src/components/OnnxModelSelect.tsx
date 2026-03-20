/**
 * ONNX Model file selection component.
 *
 * Allows the user to pick local `.onnx` and `.data` files.  Supports both
 * the legacy single-file layout (`model.onnx.data`) and the chunked layout
 * with 4-digit suffixes (`model.onnx.data.0000`, `.0001`, …).
 */

import { useCallback } from 'react';

// GitHub Releases base URL – used only to generate download links for the user.
const DEFAULT_RELEASES_URL =
  'https://github.com/KotoriK/mlsharp-web/releases/latest';

export interface OnnxModelSelectProps {
  /** Currently selected ONNX graph file (`.onnx`). */
  modelFile: File | null;
  /** Currently selected external-data files (sorted by name). */
  dataFiles: File[];
  /** Called whenever the user changes the selected model files. */
  onModelFilesChange: (onnxFile: File | null, dataFiles: File[]) => void;
  /** GitHub Releases page URL shown in the download guidance. */
  releasesUrl?: string;
}

/**
 * Standalone UI section for selecting local ONNX model files.
 *
 * The file picker accepts:
 *   - One `.onnx` graph file
 *   - One or more `.data` / `.data.XXXX` external-data files
 *
 * Data files are filtered by matching names ending with `.onnx.data` (legacy)
 * or `.onnx.data.XXXX` where XXXX is exactly four digits.  They are always
 * returned sorted by name so callers can concatenate them in order.
 */
export function OnnxModelSelect({
  modelFile,
  dataFiles,
  onModelFilesChange,
  releasesUrl = DEFAULT_RELEASES_URL,
}: OnnxModelSelectProps) {
  const handleFilesSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []);
      const onnxFile = files.find((f) => f.name.endsWith('.onnx')) ?? null;

      // Match external-data files:
      //   • chunked:  *.onnx.data.XXXX  (4-digit suffix — preferred)
      //   • legacy:   *.onnx.data       (single file, backward-compatible)
      const matched = files
        .filter((f) => /\.onnx\.data(\.\d{4})?$/.test(f.name))
        .sort((a, b) => a.name.localeCompare(b.name));

      onModelFilesChange(onnxFile, matched);
    },
    [onModelFilesChange],
  );

  return (
    <div className="model-selection">
      <h3>🤖 ONNX Model</h3>
      {modelFile ? (
        <div className="model-loaded">
          <span className="model-loaded-name">
            ✅ <strong>{modelFile.name}</strong>
            {dataFiles.length > 0 && (
              <span className="model-data-name">
                {' '}
                + {dataFiles.length === 1
                  ? dataFiles[0].name
                  : `${dataFiles.length} data files`}
              </span>
            )}
          </span>
          <label className="btn btn-secondary model-change-btn">
            Change
            <input
              type="file"
              accept=".onnx,.data"
              multiple
              onChange={handleFilesSelect}
              style={{ display: 'none' }}
            />
          </label>
        </div>
      ) : (
        <div className="model-notice">
          <p>
            Download the model files from{' '}
            <a
              href={releasesUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub Releases
            </a>
            , then select them below:
          </p>
          <ol>
            <li>
              <code>sharp_model.onnx</code> — graph file
            </li>
            <li>
              <code>sharp_model.onnx.data.0000</code>, <code>.0001</code>,
              … — external data chunks
            </li>
          </ol>
          <label className="btn btn-primary model-select-btn">
            Select Model Files
            <input
              type="file"
              accept=".onnx,.data"
              multiple
              onChange={handleFilesSelect}
              style={{ display: 'none' }}
            />
          </label>
          <p className="model-hint">
            Select the <code>.onnx</code> file together with all{' '}
            <code>.data.XXXX</code> files at once.
          </p>
          <p className="model-hint">
            You can still load existing PLY / splat files without the model.
          </p>
        </div>
      )}
    </div>
  );
}
