# Verification record

Local verification completed on 2026-09-28 using Python 3.10.20 on macOS / Apple Silicon, ONNX Runtime 1.22.0, and Playwright 1.52.0 (Chromium 136 and WebKit 18.4).

## Passed locally

- Isolated environment installation of all pinned Python teaching and test dependencies; `pip check` and imports succeeded.
- Python/JavaScript preprocessing: **bit-identical** grayscale bytes and float32 values on six deterministic RGBA arrays, including partial/full transparency, 1×1 input, non-square dimensions, downsampling and upsampling. Both browser engines passed.
- Stable softmax on large logits.
- PyTorch vs exported ONNX: eight inputs (black, midpoint, white, five seeded random inputs), `atol=1e-5`, `rtol=1e-4`; maximum absolute error **5.22×10⁻⁸** for the primary fixture.
- Actual browser WASM inference on a deterministic 137×91 PNG vs Python ONNX: both engines passed the same tolerances. Python logits `[-0.0578614511, -0.1039336324]`; browser logits `[-0.0578614324, -0.1039336324]`. Displayed scores matched Python after rounding to two decimal percentage points.
- Python JPEG EXIF orientation and HEIC decoding. Browser JPEG orientation produced the expected 100×60 oriented dimensions and matched Python within two gray levels.
- Mouse drawing, photo selection, clear, invalid-photo recovery, model replacement, invalid-model recovery, model removal, and refresh resets in Chromium and WebKit.
- Rejection of malformed files, incorrect model names/shapes/types, dynamic dimensions, external weights, nonfinite trial outputs, unsupported operations, and oversized models.
- Model/photo changes clear outdated predictions. Refresh clears selected files and image state.
- Request recording: only same-origin static GET requests, no request bodies, no selected-file uploads, no attempts to fetch external tensor data. No localStorage/sessionStorage writes. Source review found no upload, analytics or persistent-file storage path.
- All browser tests served under `/model-playground/`; runtime paths worked from that repository subpath.
- Desktop 1440px and phone emulation at 390px/320px; touch input, camera-file control selection, and 200% text reflow. Desktop and mobile screenshots inspected visually with no clipping or overlap.
- CPU synthetic training, best-checkpoint save/load, final test evaluation and O/X confusion matrix, new-image prediction, export, and checkpoint incompatibility detection.
- Automatic Apple **MPS** selection and a complete synthetic training/evaluation epoch succeeded. CPU fallback was also exercised in the restricted environment.
- Runtime downloader succeeded with pinned archive/license checksums; locally served license and third-party notices are included.

The fixture is explicitly **UNTRAINED**, outside `website/`. The training smoke test uses artificial images solely to test program flow. Neither result is evidence of recognition accuracy on student photos.

Reproduce with the commands in [README](../README.md). The detailed machine-readable report and screenshots are generated under ignored `artifacts/verification/`. GitHub Actions reruns the integration and synthetic-training checks before deployment; its run status is the authoritative CI result.

## Not verified

- Real iPhone/Android camera permissions, native file-picker behavior, HEIC decoding in mobile browsers, and actual phone memory/performance. Emulation does not establish these.
- Recognition accuracy or calibration on a labeled real-photo dataset: none was supplied, and no trained demo is published.
- Every possible student architecture/operator, every browser version, or guaranteed offline startup. A successful trial inference is a compatibility check for that model and device, not a universal guarantee.
