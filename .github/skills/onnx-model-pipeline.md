# SKILL: ONNX Model Pipeline

This document describes the end-to-end pipeline for exporting the ml-sharp
ONNX model, uploading it to GitHub Releases, and loading it in the browser.
Use it as a reference whenever you need to change **how model files are
produced** or **how the web application selects / loads them**.

---

## 1. File naming conventions

| File | Pattern | Example |
|------|---------|---------|
| ONNX graph | `<name>.onnx` | `sharp_model.onnx` |
| External data chunks | `<name>.onnx.data.XXXX` | `sharp_model.onnx.data.0000`, `.0001`, … |

> **Important:** Data files always use 4-digit zero-padded suffixes, even when
> there is only a single chunk (`.data.0000`).  This lets the web client use
> one consistent code-path for detection and reassembly.

The ONNX graph file internally references its external data by the *location*
string `<name>.onnx.data`.  The 4-digit suffixed chunks are a transport-level
split that the web client reverses by concatenating all chunks back into a
single buffer before registering it under the original location name.

---

## 2. Export script (`scripts/export_to_onnx.py`)

### What it does
1. Loads the PyTorch ml-sharp checkpoint (downloads from CDN if no path given).
2. Applies ONNX-compatibility monkey-patches (`patch_mlsharp_for_onnx`).
3. Exports to ONNX via `torch.onnx.export` with `dynamo=False` and
   `do_constant_folding=False`.
4. Converts all large tensors to external data (`onnx.save_model` with
   `save_as_external_data=True`, location = `<name>.onnx.data`).
5. Optionally simplifies with `onnxsim`.
6. Optionally verifies with ONNX Runtime (`--verify`).
7. Splits the `.onnx.data` file into numbered `.data.XXXX` chunks
   (always, unless `--chunk-size 0` is passed).

### Key CLI flags

```bash
python scripts/export_to_onnx.py \
  -o public/models/sharp_model.onnx \
  --verify \
  --chunk-size 1900000000   # default ≈ 1.9 GB per chunk
```

### When to modify
* **New model architecture / inputs / outputs** → update `SharpONNXWrapper`,
  input/output names, and dynamic axes.
* **Change chunk-size or disable splitting** → adjust `--chunk-size`.  Setting
  it to `0` disables splitting and produces a single `.onnx.data` file (legacy
  format); the website still supports this but the 4-digit format is preferred.

---

## 3. GitHub Actions workflow (`.github/workflows/export-onnx-model.yml`)

### Trigger
Manual `workflow_dispatch` with an optional `force_rebuild` boolean input.

### Flow
1. **Skip check** – probes the `model-release` tag in GitHub Releases; skips
   the export if the model already exists (unless `force_rebuild` is true).
2. **Python setup** – installs PyTorch (CPU), ONNX, ONNX Runtime, etc.
3. **Clone ml-sharp** – `pip install -e /tmp/ml-sharp`.
4. **Export** – runs `scripts/export_to_onnx.py`.
5. **Verify** – checks that `sharp_model.onnx` and at least
   `sharp_model.onnx.data.0000` exist and total size is > 1 MB.
6. **Upload to GitHub Releases** – uploads `.onnx` + all `.data.XXXX` chunks
   with `--clobber` and retry logic.
7. **Upload as workflow artifact** – 90-day retention, no compression.

### When to modify
* **New dependencies** → update the `pip install` step.
* **Different chunk naming** → update the verify step's existence check and
  the upload glob pattern.
* **New release tag or asset layout** → update the `gh release` commands and
  the skip-check probe.

---

## 4. Website file selection (`src/components/OnnxModelSelect.tsx`)

### Component: `<OnnxModelSelect />`

A self-contained React component that renders the file-picker UI and
classifies the selected files:

* **ONNX graph**: any file ending in `.onnx`.
* **Data files**: files matching `*.onnx.data.XXXX` (4-digit suffix) **or**
  `*.onnx.data` (legacy single file).  Returned sorted by name.

The parent component (`App.tsx`) receives the classified files via the
`onModelFilesChange(onnxFile, dataFiles)` callback.

### When to modify
* **Different file extensions or naming** → update the regex in
  `handleFilesSelect`:
  ```ts
  /\.onnx\.data(\.\d{4})?$/.test(f.name)
  ```
* **Additional file types** (e.g. quantization config) → extend the filter
  and add to the callback signature.

---

## 5. ONNX Runtime loading (`src/utils/onnxInference.ts`)

### URL-based loading (`fetchExternalData`)
When `modelPath` is a URL the inference engine automatically probes for
external data:

1. **Chunked** — `HEAD` request to `<model>.onnx.data.0000`; if found,
   sequentially downloads `.0000`, `.0001`, … until a `HEAD` fails, then
   concatenates all chunks into one `ArrayBuffer`.
2. **Single** — `HEAD` request to `<model>.onnx.data`; if found, passes the
   URL directly so ONNX Runtime can stream it.

### Local-file loading
When the model is loaded from a local `File` object (via `OnnxModelSelect`):

* `App.tsx` reads all selected data files, sorts them by name, and
  concatenates their `ArrayBuffer`s into a single buffer.
* The merged buffer is passed to `SharpInference` as `externalDataBuffer`,
  with `externalDataFileName` set to `<model>.onnx.data` (matching the
  internal location reference in the ONNX file).

### When to modify
* **New external-data naming scheme** → update both `fetchExternalData`
  (URL path) and the local-file concatenation logic in `App.tsx`.
* **Multiple independent external-data files** (not chunks of one file) →
  register each as a separate entry in `sessionOptions.externalData`.

---

## 6. Checklist for common changes

### Adding a new model variant
- [ ] Update `export_to_onnx.py` with new model class / wrapper
- [ ] Update GitHub Actions workflow if new dependencies are needed
- [ ] Update `OnnxModelSelect.tsx` download guidance text
- [ ] Update `onnxInference.ts` input/output tensor names

### Changing the chunk / data file format
- [ ] Update `split_external_data()` in `export_to_onnx.py`
- [ ] Update verify step in `.github/workflows/export-onnx-model.yml`
- [ ] Update upload step glob pattern in the workflow
- [ ] Update file-matching regex in `OnnxModelSelect.tsx`
- [ ] Update `fetchExternalData()` probe logic in `onnxInference.ts`
- [ ] Update buffer concatenation in `App.tsx` if needed
