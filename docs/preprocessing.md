# Preprocessing contract: ox-gray64-v1

Python and JavaScript share these steps, in this order:

1. Decode the image and respect EXIF/camera orientation. Python uses Pillow `ImageOps.exif_transpose` and registers `pillow-heif`; the browser uses `createImageBitmap` with `imageOrientation: 'from-image'`.
2. Resize the **whole oriented image** to 64×64. For destination coordinate `d` (0–63), use `floor((d + 0.5) * source_size / 64)`. Apply independently to x and y. No crop, smoothing, antialiasing or thresholding is applied by preprocessing.
3. Composite each selected RGBA pixel onto white. For each 8-bit color channel `C` and alpha `A`, use `floor((C*A + 255*(255-A) + 127) / 255)`. This fixes round-half-up behavior and avoids implementation-dependent alpha rounding.
4. Compute `gray = floor((299*R + 587*G + 114*B + 500) / 1000)` using the composited RGB values.
5. Compute `gray / 127.5 - 1` in double precision, then store as float32. Black is −1, white is +1. Tensor memory order is NCHW `[1,1,64,64]`.

The preview displays these exact integer grayscale values. It uses pixelated enlargement rather than interpolation. Display scaling never feeds back into preprocessing; photo pixels come from the full-resolution decoded source. Pointer drawings use a fixed 512×512 canvas, independent of CSS size or device pixel ratio.

This simple resize can lose thin strokes and stretches non-square images. Crop roughly square around one complete symbol **before** training or testing. Train with the same input contract; photo and finger-drawing distributions can differ substantially.

Identical RGBA arrays must produce identical grayscale and float32 arrays. Browser/Python photo decoding can still differ because of JPEG/color profiles, alpha premultiplication/unpremultiplication, and codec implementation. HEIC support varies by browser. The integration suite separates exact pixel-array checks from a JPEG orientation check (allowing ≤2 gray levels on its simple fixture).

## Model loading

The site reads ONNX protobuf metadata locally, checks exact names, element types and fixed shapes, and rejects external tensor data (including nested graphs). It then creates an ONNX Runtime Web WASM session and runs trial inference, validating finite float32 `[1,2]` output. Invalid files leave the app ready to choose another model. Model buffers are never used as URLs.

Class order is a semantic promise: neither valid tensor names nor a successful trial can prove that output 0 means O. Checkpoints include `contract_version`, `classes` and `architecture`; Python refuses incompatible checkpoints. Model file metadata is helpful for inspection, but the website does not require architecture-specific metadata so other networks remain usable.

## Runtime and export references

- [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
- [Runtime flags, WASM paths and session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)
- [PyTorch previous versions / matching torchvision releases](https://docs.pytorch.org/get-started/previous-versions/)

The example exports opset 18 with the explicitly selected legacy exporter (`dynamo=False`). This avoids external weight files and provides a small plain inference graph for this teaching example. ONNX checker and eight PyTorch/ONNX comparisons run on every export. Changing architecture may introduce unsupported operations; the browser remains the final compatibility check.
