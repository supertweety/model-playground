export const CONTRACT_VERSION = 'ox-gray64-v1';
export const CLASSES = ['O', 'X'];

// Integer round-half-up alpha compositing is part of v1, before grayscale.
export function preprocessRGBA(rgba, width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || rgba.length !== width * height * 4) {
    throw new Error('Invalid pixel array.');
  }
  const gray = new Uint8Array(4096);
  const tensor = new Float32Array(4096);
  for (let y = 0; y < 64; y++) {
    const sy = Math.floor((y + 0.5) * height / 64);
    for (let x = 0; x < 64; x++) {
      const sx = Math.floor((x + 0.5) * width / 64);
      const p = (sy * width + sx) * 4;
      const a = rgba[p + 3];
      const c = channel => Math.floor((rgba[p + channel] * a + 255 * (255 - a) + 127) / 255);
      const g = Math.floor((299 * c(0) + 587 * c(1) + 114 * c(2) + 500) / 1000);
      const i = y * 64 + x;
      gray[i] = g;
      tensor[i] = g / 127.5 - 1;
    }
  }
  return {gray, tensor};
}

export function softmax(logits) {
  if (logits.length !== 2 || !Array.from(logits).every(Number.isFinite)) throw new Error('The model returned invalid logits.');
  const max = Math.max(...logits);
  const e = Array.from(logits, value => Math.exp(value - max));
  return e.map(value => value / (e[0] + e[1]));
}
