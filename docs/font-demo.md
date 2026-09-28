# Basic-font O/X starter

This is a small, actually trained CNN provided for a quick first successful run. It recognizes **one printed uppercase O or X on a white background**, centered with margin. Tap **Try font demo** to load it and test a printed O. Choose a sample font, tap O or X, and select Run model. The model can also be downloaded to practice the phone file-picker flow.

The website draws sample text into the same canvas and uses the same preprocessing and real ONNX inference as other inputs; there are no hard-coded class scores.

## Training and evaluation

Run `python python/build_font_demo.py` from the project root after installing the Python dependencies. No font download is needed: it uses the DejaVu font files included with pinned matplotlib. Only rasterized synthetic examples are used; no student photos are collected. Font binaries are not redistributed with the website.

The generator renders DejaVu Sans, Sans Bold, Serif, Serif Bold, and Sans Mono, with varied size, position, darkness and modest rotation. It generates 1,600 training, 320 validation and 400 test renders using separate seeds. The network starts from random weights, trains for eight epochs on CPU, and selects its checkpoint by validation accuracy. The published model's complete report and SHA-256 are in [`website/demo/basic-fonts.json`](../website/demo/basic-fonts.json).

The current synthetic test result is 400/400 correct (O: 200/200; X: 200/200). These are **new renders of the same font families**, not unseen font families, handwriting, screen photographs, or real camera data. Do not interpret this score as real-world accuracy. Browser tests separately check six rendered samples (O and X in sans-serif, serif and monospace) in Chromium and WebKit.

The ONNX file is about 0.5 MiB and follows the same student contract. Export checks include eight generic tensors plus eight synthetic image tensors. Rebuilding records a new checksum; numerical results can vary with library/platform changes.

It can fail on thin strokes, lowercase letters, unfamiliar fonts, clutter, low contrast, perspective, glare, handwriting, blanks and unrelated objects. To try photographing printed text on a screen, crop tightly and roughly square around one complete large dark letter; this use has not been validated on real phones. Students should still collect their own data and train their own architecture for the assignment.
