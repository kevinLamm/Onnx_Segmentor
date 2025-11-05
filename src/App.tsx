import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

/**
 * SAM2 ONNX Image Segmenter — single-file React component
 * -------------------------------------------------------
 * What you get:
 * - Drag/drop (or click) to upload an image
 * - Click points (foreground/background) to prompt
 * - Run ONNX Runtime Web (WebGPU → WASM fallback)
 * - Renders semi‑transparent mask overlay
 * - Export mask as PNG (transparent background)
 * - Export mask as polygon (GeoJSON-like)
 *
 * Setup notes:
 * - Place your SAM2 .onnx in /public/models/sam2_tiny.onnx (or update MODEL_PATH below)
 * - This component assumes a SAM‑style single ONNX graph that accepts image tensor + prompts
 *   and returns a mask (H×W) or (N×H×W). Real models differ; adjust IO_NAMES accordingly.
 * - For best performance, enable Chrome flags for WebGPU if needed.
 */

// ======== TUNE THESE TO MATCH YOUR MODEL'S IO NAMES & PRE/POST ========
const MODEL_PATH = "/models/sam2_hiera_tiny_decoder_pr1.onnx"; // update to your model
const IO_NAMES = {
  // Inputs
  image: "image", // float32 [1,3,H,W]
  pointCoords: "point_coords", // float32 [1, P, 2] (x,y) in pixels or normalized
  pointLabels: "point_labels", // float32 [1, P] (1=fg, 0=bg)
  // Optional: box: "box_coords", // float32 [1,4] (x1,y1,x2,y2)
  // Outputs
  masks: "masks", // float32 [1,1,H,W] or [1,H,W]
  // Optional: iouScores: "iou_predictions",
};

// Normalization used by many SAM variants; adjust if your model differs
const NORM = {
  mean: [123.675, 116.28, 103.53],
  std: [58.395, 57.12, 57.375],
};

// =====================================================================

type PromptPoint = { x: number; y: number; label: 0 | 1 }; // 1=FG, 0=BG

export default function Sam2OnnxSegmenter() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [mask, setMask] = useState<ImageData | null>(null);

  const hostRef = useRef<HTMLDivElement>(null);
  const imgCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  // Load ONNX model once
  useEffect(() => {
    (async () => {
      try {
        setError(null);
        // Prefer WebGPU if available
        const providers: ort.InferenceSession.SessionOptions["executionProviders"] = [];
        //@ts-ignore
        if ("gpu" in navigator || (navigator as any).gpu) providers.push("webgpu" as any);
        providers.push("wasm");

        const so: ort.InferenceSession.SessionOptions = {
          executionProviders: providers as any,
        };
        const s = await ort.InferenceSession.create(MODEL_PATH, so);
        setSession(s);
      } catch (e: any) {
        console.error(e);
        setError("Failed to load ONNX model. Check path / browser support.");
      }
    })();
  }, []);

  // Draw image (and current mask) whenever they change
  useEffect(() => {
    draw();
  }, [img, mask]);

  const draw = () => {
    const imgCanvas = imgCanvasRef.current;
    const overlay = overlayRef.current;
    if (!img || !imgCanvas || !overlay) return;

    // Resize canvases to image
    imgCanvas.width = img.width;
    imgCanvas.height = img.height;
    overlay.width = img.width;
    overlay.height = img.height;

    const g = imgCanvas.getContext("2d")!;
    g.clearRect(0, 0, imgCanvas.width, imgCanvas.height);
    g.drawImage(img, 0, 0);

    // Mask overlay
    const og = overlay.getContext("2d")!;
    og.clearRect(0, 0, overlay.width, overlay.height);
    if (mask) {
      // Convert single-channel mask's alpha into a colored overlay
      const tmp = og.createImageData(mask.width, mask.height);
      for (let i = 0; i < mask.data.length; i += 4) {
        const a = mask.data[i + 3]; // alpha carried in mask ImageData
        tmp.data[i + 0] = 0; // R
        tmp.data[i + 1] = 255; // G
        tmp.data[i + 2] = 0; // B
        tmp.data[i + 3] = a; // A
      }
      og.putImageData(tmp, 0, 0);
    }

    // Draw points on top
    og.save();
    og.fillStyle = "#00d1ff";
    og.strokeStyle = "#003a72";
    og.lineWidth = 2;
    for (const p of points) {
      og.beginPath();
      og.arc(p.x, p.y, 5, 0, Math.PI * 2);
      og.fill();
      og.stroke();
      // label text
      og.fillStyle = p.label ? "#00ff88" : "#ff3377";
      og.font = "bold 12px ui-sans-serif";
      og.fillText(p.label ? "+" : "-", p.x + 8, p.y + 4);
      og.fillStyle = "#00d1ff";
    }
    og.restore();
  };

  // Handle file select / drop
  const onPick = (file: File) => {
    const url = URL.createObjectURL(file);
    const im = new Image();
    im.onload = () => {
      setImg(im);
      setMask(null);
      setPoints([]);
      URL.revokeObjectURL(url);
    };
    im.src = url;
  };

  const onCanvasClick = (e: React.MouseEvent) => {
    if (!imgCanvasRef.current || !img) return;
    const rect = imgCanvasRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * img.width);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * img.height);
    const fg = !(e.shiftKey || e.altKey || e.metaKey || e.ctrlKey); // hold any modifier for BG
    setPoints((pp) => [...pp, { x, y, label: fg ? 1 : 0 }]);
  };

  // Convert HTMLCanvas image to Float32 NCHW, normalized
  const canvasToTensor = (canvas: HTMLCanvasElement) => {
    const { width: W, height: H } = canvas;
    const ctx = canvas.getContext("2d")!;
    const data = ctx.getImageData(0, 0, W, H).data;
    const out = new Float32Array(3 * H * W);
    let i = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        const r = data[idx + 0];
        const g = data[idx + 1];
        const b = data[idx + 2];
        out[i + 0 * H * W] = (r - NORM.mean[0]) / NORM.std[0];
        out[i + 1 * H * W] = (g - NORM.mean[1]) / NORM.std[1];
        out[i + 2 * H * W] = (b - NORM.mean[2]) / NORM.std[2];
        i++;
      }
    }
    return { tensor: out, H, W };
  };

  const run = async () => {
    if (!session || !img || !imgCanvasRef.current) return;
    try {
      setBusy(true);
      setError(null);

      const { tensor, H, W } = canvasToTensor(imgCanvasRef.current);

      // Build ONNX feeds — adjust to your graph's expected shapes/normalization
      const image = new ort.Tensor("float32", tensor, [1, 3, H, W]);

      // Points: SAM variants expect either absolute pixels or normalized [0..1].
      // Here we provide absolute pixels; adjust if your model expects normalized.
      const P = Math.max(points.length, 1);
      const coords = new Float32Array(1 * P * 2);
      const labels = new Float32Array(1 * P);
      for (let i = 0; i < P; i++) {
        const p = points[i] ?? { x: 0, y: 0, label: 0 as 0 | 1 };
        coords[i * 2 + 0] = p.x; // or p.x / W
        coords[i * 2 + 1] = p.y; // or p.y / H
        labels[i] = p.label;
      }
      const pointCoords = new ort.Tensor("float32", coords, [1, P, 2]);
      const pointLabels = new ort.Tensor("float32", labels, [1, P]);

      const feeds: Record<string, ort.Tensor> = {
        [IO_NAMES.image]: image,
        [IO_NAMES.pointCoords]: pointCoords,
        [IO_NAMES.pointLabels]: pointLabels,
      };

      const out = await session.run(feeds);

      // Retrieve mask; support [1,1,H,W] or [1,H,W]
      const m = out[IO_NAMES.masks];
      if (!m) throw new Error("Output mask not found. Check IO_NAMES.masks");

      let maskData: Float32Array;
      let shape = m.dims;
      if (shape.length == 4) {
        maskData = m.data as Float32Array; // [1,1,H,W]
        // strip batch & channel
        shape = [shape[2], shape[3]];
      } else if (shape.length == 3) {
        maskData = m.data as Float32Array; // [1,H,W]
        shape = [shape[1], shape[2]];
      } else if (shape.length == 2) {
        maskData = m.data as Float32Array; // [H,W]
      } else {
        throw new Error(`Unexpected mask shape ${JSON.stringify(m.dims)}`);
      }

      const [MH, MW] = shape as [number, number];
      // If model returns resized mask, optionally resample to image size
      const imgW = W, imgH = H;
      const maskImg = new ImageData(imgW, imgH);

      // Simple nearest-neighbor upscale if needed
      for (let y = 0; y < imgH; y++) {
        for (let x = 0; x < imgW; x++) {
          const srcX = Math.floor((x / imgW) * MW);
          const srcY = Math.floor((y / imgH) * MH);
          const v = (m.data as Float32Array)[srcY * MW + srcX]; // assume single mask
          const alpha = Math.max(0, Math.min(255, Math.round((v > 0.0 ? 1 : 0) * 110)));
          const idx = (y * imgW + x) * 4;
          maskImg.data[idx + 0] = 0;
          maskImg.data[idx + 1] = 0;
          maskImg.data[idx + 2] = 0;
          maskImg.data[idx + 3] = alpha; // store alpha only; colorized at draw()
        }
      }

      setMask(maskImg);
    } catch (e: any) {
      console.error(e);
      setError(e.message ?? String(e));
    } finally {
      setBusy(false);
    }
  };

  const clearPoints = () => setPoints([]);
  const undoPoint = () => setPoints((pp) => pp.slice(0, -1));

  const exportPNG = () => {
    if (!overlayRef.current || !imgCanvasRef.current) return;
    // Compose base image + mask overlay into one PNG with transparent background for non-mask areas
    const W = imgCanvasRef.current.width;
    const H = imgCanvasRef.current.height;
    const out = document.createElement("canvas");
    out.width = W;
    out.height = H;
    const g = out.getContext("2d")!;
    // Start fully transparent, then draw mask as opaque white on alpha
    const maskCtx = overlayRef.current.getContext("2d")!;
    const maskData = maskCtx.getImageData(0, 0, W, H);
    const bin = new ImageData(W, H);
    for (let i = 0; i < maskData.data.length; i += 4) {
      const a = maskData.data[i + 3];
      const on = a > 0;
      bin.data[i + 0] = 255;
      bin.data[i + 1] = 255;
      bin.data[i + 2] = 255;
      bin.data[i + 3] = on ? 255 : 0;
    }
    g.putImageData(bin, 0, 0);

    const link = document.createElement("a");
    link.download = "mask.png";
    link.href = out.toDataURL("image/png");
    link.click();
  };

  // Marching Squares to polygonize mask
  const exportPolygon = () => {
    if (!overlayRef.current) return;
    const W = overlayRef.current.width;
    const H = overlayRef.current.height;
    const ctx = overlayRef.current.getContext("2d")!;
    const id = ctx.getImageData(0, 0, W, H);
    const isOn = (x: number, y: number) => id.data[(y * W + x) * 4 + 3] > 0;

    // Basic border trace (clockwise) for the largest blob
    const visited = new Uint8Array(W * H);
    let sx = -1, sy = -1;
    for (let y = 0; y < H && sy === -1; y++) {
      for (let x = 0; x < W; x++) {
        if (isOn(x, y)) { sx = x; sy = y; break; }
      }
    }
    if (sx === -1) return alert("No mask to export");

    const poly: Array<[number, number]> = [];
    let x = sx, y = sy, dir = 0; // 0=R,1=D,2=L,3=U
    for (let steps = 0; steps < W * H * 4; steps++) {
      poly.push([x, y]);
      visited[y * W + x] = 1;
      // Right-hand rule along boundary
      const right = (dir + 3) & 3, left = (dir + 1) & 3;
      const move = (d: number) => {
        if (d === 0) x++; else if (d === 1) y++; else if (d === 2) x--; else y--;
      };
      const ahead = () => (dir === 0 ? isOn(x + 1, y) : dir === 1 ? isOn(x, y + 1) : dir === 2 ? isOn(x - 1, y) : isOn(x, y - 1));
      const rightIsOn = right === 0 ? isOn(x + 1, y) : right === 1 ? isOn(x, y + 1) : right === 2 ? isOn(x - 1, y) : isOn(x, y - 1);
      if (rightIsOn) { dir = right; move(dir); }
      else if (ahead()) { move(dir); }
      else { dir = left; }
      if (x === sx && y === sy && poly.length > 10) break;
    }

    const gj = {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: {},
          geometry: { type: "Polygon", coordinates: [poly.map(([px, py]) => [px, py])] },
        },
      ],
    } as const;

    const blob = new Blob([JSON.stringify(gj)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mask.geojson";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-start p-4 gap-4 bg-neutral-50">
      <div className="w-full max-w-5xl grid md:grid-cols-[1fr,320px] gap-4">
        <div className="bg-white rounded-2xl shadow p-3 flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">SAM2 ONNX Image Segmenter</h1>
            <div className="text-sm opacity-70">{session ? "Model ready" : "Loading model…"}</div>
          </div>
          <div
            ref={hostRef}
            className="relative w-full aspect-[4/3] bg-neutral-100 rounded-xl overflow-hidden border border-neutral-200"
            onContextMenu={(e) => e.preventDefault()}
          >
            {img ? (
              <>
                <canvas
                  ref={imgCanvasRef}
                  className="absolute inset-0 w-full h-full object-contain"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={onCanvasClick}
                />
                <canvas
                  ref={overlayRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                />
              </>
            ) : (
              <Dropzone onPick={onPick} />
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              className="px-3 py-2 bg-black text-white rounded-xl disabled:opacity-50"
              disabled={!session || !img || busy}
              onClick={run}
            >
              {busy ? "Running…" : "Run segmentation"}
            </button>
            <button className="px-3 py-2 bg-neutral-200 rounded-xl" onClick={undoPoint} disabled={!points.length}>
              Undo point
            </button>
            <button className="px-3 py-2 bg-neutral-200 rounded-xl" onClick={clearPoints} disabled={!points.length}>
              Clear points
            </button>
            <button className="px-3 py-2 bg-neutral-200 rounded-xl" onClick={exportPNG} disabled={!mask}>
              Export mask PNG
            </button>
            <button className="px-3 py-2 bg-neutral-200 rounded-xl" onClick={exportPolygon} disabled={!mask}>
              Export polygon
            </button>
            <label className="ml-auto text-sm opacity-70">Hint: click to add FG point; hold Shift/Alt/Ctrl for BG</label>
          </div>
          {error && (
            <div className="text-red-600 text-sm bg-red-50 border border-red-200 rounded-md p-2">{error}</div>
          )}
        </div>

        <aside className="bg-white rounded-2xl shadow p-4 flex flex-col gap-3">
          <h2 className="font-semibold">How to use</h2>
          <ol className="list-decimal list-inside text-sm space-y-1">
            <li>Drop an image or click to upload.</li>
            <li>Click the image to add <b>foreground</b> points. Hold Shift/Alt/Ctrl while clicking to add <b>background</b> points.</li>
            <li>Hit <b>Run segmentation</b>. Tweak points and re-run if needed.</li>
            <li>Export as transparent PNG or polygon.</li>
          </ol>
          <h2 className="font-semibold mt-2">Model I/O mapping</h2>
          <p className="text-sm">
            Update <code>MODEL_PATH</code>, <code>IO_NAMES</code>, and the normalization constants if your SAM2 ONNX uses different names or expects normalized coordinates.
          </p>
          <div className="text-xs text-neutral-600">
            Coordinates are sent in <b>pixels</b>; change to normalized 0..1 if needed.
          </div>
          <h2 className="font-semibold mt-2">Performance tips</h2>
          <ul className="list-disc list-inside text-sm space-y-1">
            <li>Use <b>onnxruntime-web</b> with the <b>WebGPU</b> execution provider for big speedups.</li>
            <li>Resize very large images client‑side (e.g., to 1024px max side) before inference.</li>
            <li>Quantized (int8) ONNX can help on WASM; verify accuracy.</li>
          </ul>
        </aside>
      </div>
    </div>
  );
}

function Dropzone({ onPick }: { onPick: (f: File) => void }) {
  const [hover, setHover] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <div
      className={`absolute inset-0 flex flex-col items-center justify-center gap-2 ${
        hover ? "bg-neutral-200" : "bg-neutral-100"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setHover(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onPick(f);
      }}
    >
      <div className="text-sm opacity-70">Drop image here</div>
      <button
        className="px-3 py-2 bg-neutral-900 text-white rounded-xl"
        onClick={() => inputRef.current?.click()}
      >
        Select image…
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onPick(f);
        }}
      />
    </div>
  );
}
