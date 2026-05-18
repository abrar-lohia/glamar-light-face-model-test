import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';

const MODEL_BASE = './best_128_light_128_model';
const lightModelDef = await fetch(`${MODEL_BASE}/glamar-light-detection-model.json`).then(r => r.json());
const lightModelShardUrl = `${MODEL_BASE}/glamar-light-detection-model-shard1of1.bin`;

// ── Config ───────────────────────────────────────────────────────────
const LIGHT_INPUT = 128;
const CLASS_NAMES = ['Low', 'bright', 'darkbright', 'normal'];
const CLASS_KEYS = ['low', 'bright', 'darkbright', 'normal'];
const BAR_COLORS = ['#ef4444', '#facc15', '#a78bfa', '#22c55e'];
const CONFIDENCE_THRESHOLD = 0.4;
const STABLE_FRAMES = 4;
const PERF_INTERVAL_MS = 12000;
const WEBHOOK_URL = 'https://webhook.site/0607823d-b31f-470f-a4b5-824646924b8e';

let CANVAS_W = 480;
let CANVAS_H = 360;

// ── Event Tracking ───────────────────────────────────────────────────
const SESSION_ID = crypto.randomUUID();
const sessionStartTime = Date.now();
const classDurations = { low: 0, bright: 0, darkbright: 0, normal: 0 };
let noFaceDuration = 0;
let lastClassTimestamp = Date.now();
let lastTrackedClass = null;
let lastFaceCount = -1;
let perfSamples = [];
let lastPerfFlush = Date.now();

function emit(eventName, payload) {
  const envelope = {
    session_id: SESSION_ID,
    event: eventName,
    ts: Date.now(),
    iso: new Date().toISOString(),
    payload,
  };
  console.log(`[event] ${eventName}`, envelope);
  try {
    navigator.sendBeacon(WEBHOOK_URL, JSON.stringify(envelope));
  } catch (_) {
    fetch(WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(envelope),
      keepalive: true,
    }).catch(() => {});
  }
}

function updateClassDuration() {
  const now = Date.now();
  const elapsed = now - lastClassTimestamp;
  if (lastTrackedClass != null) {
    classDurations[CLASS_KEYS[lastTrackedClass]] += elapsed;
  }
  lastClassTimestamp = now;
}

function updateNoFaceDuration(faceCount) {
  const now = Date.now();
  if (lastFaceCount === 0) {
    noFaceDuration += now - lastClassTimestamp;
  }
}

emit('session_start', {
  model_version: 'v4.15.0',
  ua: navigator.userAgent,
  viewport: { w: window.innerWidth, h: window.innerHeight },
  referrer: document.referrer || '',
});

window.addEventListener('beforeunload', () => {
  updateClassDuration();
  emit('session_end', {
    duration_ms: Date.now() - sessionStartTime,
    class_durations_ms: { ...classDurations },
    no_face_duration_ms: noFaceDuration,
  });
});

// ── DOM ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('canvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const statusEl = document.getElementById('status');
const webcamVideo = document.getElementById('webcamVideo');

const ctx = canvas.getContext('2d');
const overlayCtx = overlayCanvas.getContext('2d');

function resizeCanvas(videoW, videoH) {
  const maxW = Math.min(window.innerWidth - 24, 640);
  const aspect = videoH / videoW;

  CANVAS_W = Math.round(maxW);
  CANVAS_H = Math.round(maxW * aspect);

  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  overlayCanvas.width = CANVAS_W;
  overlayCanvas.height = CANVAS_H;
  canvas.style.width = CANVAS_W + 'px';
  canvas.style.height = CANVAS_H + 'px';
  overlayCanvas.style.width = CANVAS_W + 'px';
  overlayCanvas.style.height = CANVAS_H + 'px';
}
const drawingUtils = new DrawingUtils(overlayCtx);

function getOrCreate(id) {
  let el = document.getElementById(id);
  if (!el) { el = document.createElement('div'); el.id = id; document.body.appendChild(el); }
  return el;
}
const resultsEl = getOrCreate('results');
const timingEl = getOrCreate('timing');
const faceInfoEl = getOrCreate('faceInfo');

// ── State ────────────────────────────────────────────────────────────
let lightModel = null;
let faceLandmarker = null;
let rafId = null;
let lastClass = null;
let streak = 0;
let lastVideoTime = -1;
let currentFrameCount = 0;
const FACE_SKIP_RATE = 4;
let cachedFaceResult = null;

function setStatus(msg, type) {
  statusEl.textContent = msg;
  statusEl.className = type || '';
}

// ── Load Models ──────────────────────────────────────────────────────
async function loadModels() {
  try {
    setStatus('Loading light detection model…');
    const shardResp = await fetch(lightModelShardUrl);
    if (!shardResp.ok) {
      throw new Error(`Failed to fetch model shard: ${shardResp.status} ${shardResp.statusText}`);
    }
    const weightData = await shardResp.arrayBuffer();
    const weightSpecs = (lightModelDef.weightsManifest || []).flatMap((group) => group.weights || []);
    lightModel = await tf.loadGraphModel(tf.io.fromMemory({
      modelTopology: lightModelDef.modelTopology,
      weightSpecs,
      weightData,
      format: lightModelDef.format,
      generatedBy: lightModelDef.generatedBy,
      convertedBy: lightModelDef.convertedBy,
      signature: lightModelDef.signature,
    }));
    const dummy = tf.zeros([1, LIGHT_INPUT, LIGHT_INPUT, 3]);
    lightModel.predict(dummy).dispose();
    dummy.dispose();

    setStatus('Loading face landmark model…');
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });

    setStatus('Models ready. Starting camera…', 'success');
    startWebcam();
  } catch (err) {
    console.error(err);
    emit('error', { where: 'model_load', message: err.message, stack: err.stack || null });
    setStatus('Failed to load models. See console.', 'error');
  }
}

// ── Light Detection Inference ────────────────────────────────────────
function inferLight() {
  const input = tf.tidy(() => {
    let img = tf.browser.fromPixels(canvas).toFloat().div(255);
    img = tf.image.resizeBilinear(img.expandDims(0), [LIGHT_INPUT, LIGHT_INPUT]);
    return img;
  });

  const rawOutput = lightModel.predict(input);
  const output = Array.isArray(rawOutput) ? rawOutput[0] : rawOutput;
  const probs = Array.from(output.dataSync());

  input.dispose();
  if (Array.isArray(rawOutput)) rawOutput.forEach(t => t.dispose());
  else rawOutput.dispose();

  const best = probs.indexOf(Math.max(...probs));
  const conf = probs[best];

  if (conf >= CONFIDENCE_THRESHOLD && best === lastClass) {
    streak++;
  } else if (conf >= CONFIDENCE_THRESHOLD) {
    lastClass = best;
    streak = 1;
  } else {
    lastClass = null;
    streak = 0;
  }

  return { probs, best, conf, stable: streak >= STABLE_FRAMES };
}

// ── Face Landmark Inference ──────────────────────────────────────────
function inferFaceVideo(source, ts) {
  if (!faceLandmarker) return null;
  return faceLandmarker.detectForVideo(source, ts);
}

// ── Drawing ──────────────────────────────────────────────────────────
const FACE_CONNECTIONS = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
const FACE_CONTOURS = FaceLandmarker.FACE_LANDMARKS_FACE_OVAL;
const FACE_IRISES = FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS.concat(
  FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS
);

function drawFaceLandmarks(result) {
  overlayCtx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  if (!result || !result.faceLandmarks || result.faceLandmarks.length === 0) return;

  for (const landmarks of result.faceLandmarks) {
    drawingUtils.drawConnectors(landmarks, FACE_CONNECTIONS, {
      color: '#30363d', lineWidth: 0.5,
    });
    drawingUtils.drawConnectors(landmarks, FACE_CONTOURS, {
      color: '#38bdf8', lineWidth: 1.5,
    });
    drawingUtils.drawConnectors(landmarks, FACE_IRISES, {
      color: '#a78bfa', lineWidth: 1.5,
    });
  }
}

function renderBars(probs) {
  resultsEl.innerHTML = probs
    .map((p, i) => {
      const pct = (p * 100).toFixed(1);
      return `<div class="bar-row">
        <span class="bar-label">${CLASS_NAMES[i]}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%;background:${BAR_COLORS[i]}"></div>
        </div>
        <span class="bar-pct">${pct}%</span>
      </div>`;
    })
    .join('');
}

function drawLightOverlay(label, confidence, stable) {
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.fillRect(0, 0, CANVAS_W, 28);
  ctx.font = 'bold 14px system-ui, sans-serif';
  ctx.fillStyle = stable ? '#22c55e' : '#facc15';
  const tag = stable ? 'STABLE' : `${streak}/${STABLE_FRAMES}`;
  ctx.fillText(`Light: ${label}  ${(confidence * 100).toFixed(1)}%  [${tag}]`, 8, 19);
  ctx.restore();
}

// ── Webcam ───────────────────────────────────────────────────────────
async function startWebcam() {
  emit('camera_state_change', { state: 'requested', error: null });
  try {
    const isMobile = /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
    const constraints = isMobile
      ? { video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 640 } } }
      : { video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    webcamVideo.srcObject = stream;
    await webcamVideo.play();

    const videoW = webcamVideo.videoWidth;
    const videoH = webcamVideo.videoHeight;
    resizeCanvas(videoW, videoH);

    lastClass = null;
    streak = 0;
    lastVideoTime = -1;
    currentFrameCount = 0;
    cachedFaceResult = null;
    emit('camera_state_change', { state: 'running', error: null });
    setStatus('Camera running.', 'success');
    rafId = requestAnimationFrame(webcamLoop);
  } catch (err) {
    console.error(err);
    const errName = err.name || err.message;
    const state = errName === 'NotAllowedError' ? 'denied' : 'error';
    emit('camera_state_change', { state, error: errName });
    emit('error', { where: 'camera', message: err.message, stack: err.stack || null });
    setStatus(`Could not access camera: ${errName}`, 'error');
  }
}

// ── Main Loop ────────────────────────────────────────────────────────
function webcamLoop() {
  const now = performance.now();
  const t0 = performance.now();

  ctx.drawImage(webcamVideo, 0, 0, CANVAS_W, CANVAS_H);

  let lightResult;
  try {
    lightResult = inferLight();
  } catch (err) {
    emit('error', { where: 'inference', message: err.message, stack: err.stack || null });
    rafId = requestAnimationFrame(webcamLoop);
    return;
  }

  if (currentFrameCount % FACE_SKIP_RATE === 0) {
    if (webcamVideo.currentTime !== lastVideoTime) {
      lastVideoTime = webcamVideo.currentTime;
      cachedFaceResult = inferFaceVideo(webcamVideo, now);
    }
  }
  currentFrameCount++;

  const totalMs = (performance.now() - t0).toFixed(1);
  const fps = (1000 / parseFloat(totalMs)).toFixed(1);

  renderBars(lightResult.probs);
  drawLightOverlay(CLASS_NAMES[lightResult.best], lightResult.conf, lightResult.stable);
  drawFaceLandmarks(cachedFaceResult);

  const numFaces = cachedFaceResult?.faceLandmarks?.length ?? 0;
  faceInfoEl.textContent = `Faces detected: ${numFaces}`;
  timingEl.textContent = `Total: ${totalMs} ms  |  ${fps} FPS`;

  // ── Event: light_classification_change (only on class flip) ──────
  const currentStableClass = lightResult.stable ? lightResult.best : null;
  if (currentStableClass !== null && currentStableClass !== lastTrackedClass) {
    const dist = {};
    lightResult.probs.forEach((p, i) => { dist[CLASS_KEYS[i]] = parseFloat(p.toFixed(4)); });
    emit('light_classification_change', {
      from: lastTrackedClass != null ? CLASS_KEYS[lastTrackedClass] : null,
      to: CLASS_KEYS[currentStableClass],
      stable: true,
      distribution: dist,
      faces_detected: numFaces,
    });
    updateClassDuration();
    lastTrackedClass = currentStableClass;
  }

  // ── Event: face_detection_change (only on count change) ──────────
  if (numFaces !== lastFaceCount && lastFaceCount !== -1) {
    updateNoFaceDuration(numFaces);
    emit('face_detection_change', {
      faces_detected: numFaces,
      previous_count: lastFaceCount,
    });
  }
  lastFaceCount = numFaces;

  // ── Event: perf_snapshot (periodic) ──────────────────────────────
  perfSamples.push({ fps: parseFloat(fps), inferMs: parseFloat(totalMs) });
  if (Date.now() - lastPerfFlush >= PERF_INTERVAL_MS && perfSamples.length > 0) {
    const fpsArr = perfSamples.map(s => s.fps);
    const msArr = perfSamples.map(s => s.inferMs);
    emit('perf_snapshot', {
      avg_fps: parseFloat((fpsArr.reduce((a, b) => a + b, 0) / fpsArr.length).toFixed(1)),
      avg_inference_ms: parseFloat((msArr.reduce((a, b) => a + b, 0) / msArr.length).toFixed(1)),
      min_fps: parseFloat(Math.min(...fpsArr).toFixed(1)),
      max_inference_ms: parseFloat(Math.max(...msArr).toFixed(1)),
      sample_count: perfSamples.length,
    });
    perfSamples = [];
    lastPerfFlush = Date.now();
  }

  rafId = requestAnimationFrame(webcamLoop);
}

// ── Boot ─────────────────────────────────────────────────────────────
loadModels();
