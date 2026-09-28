import {preprocessRGBA, softmax} from './preprocess.js';
import {validateContract} from './contract.js';
const $ = id => document.getElementById(id);
const draw = $('drawing'), ctx = draw.getContext('2d', {willReadFrequently: true});
const preview = $('processed').getContext('2d');
let session = null, ort = null, runtimePromise = null, input = null;
let modelGeneration = 0, imageGeneration = 0, revision = 0, running = false, loading = false;
let pointer = null, photoMode = false;
let activeRun = Promise.resolve();

function status(id, message, error = false) {
  $(id).textContent = message;
  $(id).classList.toggle('error', error);
}
function controls() {
  $('run').disabled = !session || !input || running || loading;
  $('reset-model').disabled = !session && !loading;
}
function invalidate() {
  revision++;
  $('prediction').textContent = '—';
  for (const c of ['o', 'x']) { $(`score-${c}`).textContent = '—'; $(`bar-${c}`).style.width = '0%'; }
  status('run-status', session && input ? 'Ready when you are.' : 'Load a model and add an image to run.');
  controls();
}
function blankCanvas() {
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 512, 512);
}
function updateInput(rgba, width, height) {
  input = preprocessRGBA(rgba, width, height);
  const pixels = new ImageData(64, 64);
  input.gray.forEach((g, i) => { pixels.data.set([g, g, g, 255], i * 4); });
  preview.putImageData(pixels, 0, 0);
  $('drawing-hint').hidden = true;
  invalidate();
}
function clearImage() {
  imageGeneration++; pointer = null; photoMode = false; input = null;
  blankCanvas(); preview.clearRect(0, 0, 64, 64);
  $('drawing-hint').hidden = false;
  $('draw-mode').classList.add('active'); $('draw-mode').setAttribute('aria-pressed', 'true');
  $('photo-file').value = ''; $('camera-file').value = '';
  status('image-status', 'Use your finger or mouse. Dark strokes work best.');
  invalidate();
}
async function runtime() {
  if (!runtimePromise) runtimePromise = import('./vendor/ort/ort.wasm.min.mjs').then(module => {
    module.env.wasm.wasmPaths = new URL('./vendor/ort/', import.meta.url).href;
    module.env.wasm.numThreads = 1;
    module.env.wasm.proxy = false;
    return module;
  }).catch(() => { runtimePromise = null; throw new Error('The runtime could not load. Check the connection; the teacher may need to prepare runtime files.'); });
  return runtimePromise;
}
async function release(old) {
  if (old) { await activeRun.catch(() => {}); await old.release().catch(() => {}); }
}
function checkOutput(result) {
  if (Object.keys(result).length !== 1 || !result.logits || result.logits.type !== 'float32' || result.logits.dims.join(',') !== '1,2') {
    throw new Error('Expected one float32 output named logits with shape [1, 2].');
  }
  return softmax(result.logits.data);
}
$('model-file').addEventListener('change', async event => {
  const file = event.target.files[0]; event.target.value = '';
  if (!file) return;
  const generation = ++modelGeneration;
  const old = session; session = null; loading = true; invalidate(); controls();
  status('model-status', `Checking ${file.name}…`);
  let candidate;
  try {
    await release(old);
    if (!file.name.toLowerCase().endsWith('.onnx')) throw new Error('Choose a .onnx file.');
    if (!file.size || file.size > 32 * 1024 * 1024) throw new Error('Choose a nonempty ONNX file up to 32 MB (32 MiB).');
    const buffer = await file.arrayBuffer();
    validateContract(buffer);
    ort = await runtime();
    if (generation !== modelGeneration) return;
    candidate = await ort.InferenceSession.create(buffer, {executionProviders: ['wasm']});
    if (candidate.inputNames.join(',') !== 'image' || candidate.outputNames.join(',') !== 'logits') throw new Error('Expected input image and output logits only.');
    const trial = await candidate.run({image: new ort.Tensor('float32', new Float32Array(4096), [1, 1, 64, 64])});
    try { checkOutput(trial); } finally { Object.values(trial).forEach(t => t.dispose()); }
    if (generation !== modelGeneration) return;
    session = candidate; candidate = null;
    status('model-status', `${file.name} · Ready · O, X`);
  } catch (error) {
    if (generation === modelGeneration) status('model-status', `Could not load model. ${error.message} Check the contract, embedded weights, and browser-supported operators; then choose another file.`, true);
  } finally {
    if (candidate) await candidate.release().catch(() => {});
    if (generation === modelGeneration) { loading = false; invalidate(); controls(); }
  }
});
$('reset-model').addEventListener('click', () => {
  modelGeneration++; loading = false;
  const old = session; session = null; void release(old);
  status('model-status', 'Choose an ONNX file to begin. Up to 32 MB.'); invalidate();
});
async function selectPhoto(event) {
  const file = event.target.files[0]; event.target.value = '';
  if (!file) return;
  clearImage(); const generation = imageGeneration;
  status('image-status', 'Opening photo…');
  let bitmap;
  try {
    if (file.size > 40 * 1024 * 1024) throw new Error('Choose a photo smaller than 40 MB.');
    bitmap = await createImageBitmap(file, {imageOrientation: 'from-image', premultiplyAlpha: 'none'});
    if (generation !== imageGeneration) return;
    if (bitmap.width * bitmap.height > 40_000_000) throw new Error('Choose a photo with fewer than 40 million pixels.');
    const source = document.createElement('canvas'); source.width = bitmap.width; source.height = bitmap.height;
    const sc = source.getContext('2d', {willReadFrequently: true}); sc.drawImage(bitmap, 0, 0);
    const pixels = sc.getImageData(0, 0, source.width, source.height);
    updateInput(pixels.data, pixels.width, pixels.height);
    blankCanvas(); ctx.drawImage(bitmap, 0, 0, 512, 512);
    source.width = source.height = 0;
    photoMode = true; $('draw-mode').classList.remove('active'); $('draw-mode').setAttribute('aria-pressed', 'false');
    status('image-status', `${file.name} · ${bitmap.width} × ${bitmap.height}. Choose Draw for a fresh canvas.`);
  } catch (error) {
    if (generation === imageGeneration) status('image-status', `Could not open photo. ${error.message} Try JPEG or PNG; HEIC support varies by browser.`, true);
  } finally { bitmap?.close(); }
}
$('photo-file').addEventListener('change', selectPhoto);
$('camera-file').addEventListener('change', selectPhoto);
$('clear').addEventListener('click', clearImage);
$('draw-mode').addEventListener('click', clearImage);
function point(event) { const r = draw.getBoundingClientRect(); return [(event.clientX - r.left) * 512 / r.width, (event.clientY - r.top) * 512 / r.height]; }
draw.addEventListener('pointerdown', event => {
  if (pointer !== null || (event.pointerType === 'mouse' && event.button !== 0)) return;
  event.preventDefault(); if (photoMode) clearImage();
  imageGeneration++; pointer = event.pointerId; draw.setPointerCapture(pointer);
  const [x, y] = point(event); ctx.strokeStyle = '#111'; ctx.fillStyle = '#111'; ctx.lineWidth = 22; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.moveTo(x, y);
  updateInput(ctx.getImageData(0, 0, 512, 512).data, 512, 512);
  status('image-status', 'Drawing · One complete O or X per image.');
});
draw.addEventListener('pointermove', event => {
  if (event.pointerId !== pointer) return;
  const [x, y] = point(event); ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
  updateInput(ctx.getImageData(0, 0, 512, 512).data, 512, 512);
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) draw.addEventListener(type, event => { if (event.pointerId === pointer) pointer = null; });
$('run').addEventListener('click', async () => {
  if (!session || !input || running) return;
  const current = revision, selected = session;
  running = true; controls(); status('run-status', 'Running on your device…');
  const tensor = new ort.Tensor('float32', input.tensor, [1, 1, 64, 64]);
  try {
    activeRun = selected.run({image: tensor});
    const result = await activeRun;
    try {
      const scores = checkOutput(result);
      if (revision !== current) return;
      $('prediction').textContent = scores[0] === scores[1] ? 'Tie' : scores[0] > scores[1] ? 'O' : 'X';
      for (const [i, c] of ['o', 'x'].entries()) { $(`score-${c}`).textContent = `${(scores[i] * 100).toFixed(2)}%`; $(`bar-${c}`).style.width = `${scores[i] * 100}%`; }
      status('run-status', 'Complete · Scores in O, X order.');
    } finally { Object.values(result).forEach(t => t.dispose()); }
  } catch (error) { if (revision === current) status('run-status', `Inference failed: ${error.message} Try another model or image.`, true); }
  finally { tensor.dispose(); running = false; controls(); }
});
clearImage();
