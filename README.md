# SAM2 ONNX Image Segmenter (100% in-browser)

This is a minimal React + Vite app that runs a SAM2-style ONNX model directly in the browser using **onnxruntime-web** (WebGPU with WASM fallback).

## Quick start

```bash
npm i
npm run dev
```

> Make sure you put your model at `public/models/sam2_tiny.onnx` (or change `MODEL_PATH` in `src/App.tsx`).

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — SAM2 ONNX web segmenter"
git branch -M main
git remote add origin https://github.com/<your-username>/sam2-onnx-web-segmenter.git
git push -u origin main
```

## Notes

- If your ONNX export uses different input/output names or expects normalized coordinates, update `IO_NAMES` and the pre/post-processing in `src/App.tsx`.
- WebGPU gives the best performance when available; otherwise WASM is used automatically.
