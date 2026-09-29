# Student guide

Open the [live Model Playground](https://supertweety.github.io/model-playground/) on your phone or computer. The website is ready to use; there is nothing to install or serve locally. Tap **Try font demo** for a first run.

For the assignment, train your network in Python, [export it to ONNX](export-tutorial.md), then select that file in the live app. The [phone guide](phone-guide.md) explains how to transfer and find it.

## 1. Collect a small pilot, then expand

Start with perhaps 20–30 distinct drawings per class to check the entire workflow. An initial larger target is **200–500 distinct examples per class**. This is a starting point, not a guarantee of accuracy.

Photograph one complete handwritten **O** or **X** per image. Crop roughly square with some margin around every stroke. Vary writers, pens, line thickness, paper/background, lighting, shadows and camera angle. Preserve the distinction between O and X; do not include blank cells or whole boards. Avoid very thin strokes: resizing to 64×64 may remove them.

Several photographs of one physical drawing are not independent examples. Assign a group identifier to each drawing/sheet/capture session before splitting. Keep **every photo from the same physical drawing, sheet, or capture session in the same split**. If groups overlap, keep the connected group together. Otherwise a model can appear accurate by remembering paper or handwriting it already saw.

Aim for **70% training, 15% validation, 15% test** while respecting groups and representing both classes in every split. Do not randomly split individual files from the same group. Keep the test set untouched until your final evaluation; use validation for tuning. If you tune after seeing test results, you need a fresh held-out test set.

## 2. Prepare your Python training environment and data

Use your teacher's Python notebook/environment if one is provided. For Colab or another hosted notebook, follow the [notebook setup](export-tutorial.md#google-colab-or-another-hosted-notebook). Python setup is for training and exporting; use the live app for browser testing.

If you want to run the Python example on your own computer, download this repository (GitHub **Code → Download ZIP**, then extract it) or clone it. Open a terminal in its root folder. Use Python 3.10–3.12 (verified with 3.10 on Apple Silicon):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements.txt
python -m pip check
```

On Windows, use `python` instead of `python3` and activate with `.venv\Scripts\activate`. Keep the pinned dependencies in this isolated environment. No runtime download or web server is required.

With the training environment ready, run from the repository root:

```sh
python python/setup_data.py
```

Put labeled photos into:

```text
data/
  train/O/         train/X/
  validation/O/    validation/X/
  test/O/          test/X/
```

The script creates empty folders only. It does not collect photos, label them, or decide group splits. JPEG, PNG, HEIC and HEIF are supported by Python. EXIF orientation is applied before preprocessing.

## 3. Train a network

```sh
python python/train.py --epochs 20
```

The small CNN in `python/model.py` starts with random weights. It uses cross-entropy on raw logits, modest training-only rotation (±12°), translation (±6%) and scale (0.9–1.1), and Adam. Validation and test images have no augmentation.

The best validation-accuracy checkpoint is saved as `checkpoints/best.pt`. The test set is evaluated once after selecting that checkpoint. The printed O/X confusion matrix has **true class in rows and predicted class in columns**, both ordered O then X. Off-diagonal values count mistakes.

Apple MPS is selected when available; otherwise the example uses CPU. For an unsupported MPS operation or a CPU comparison, run `python python/train.py --device cpu`. Seeds help repeat experiments but do not promise identical GPU results on every device.

To design your own architecture, edit `Net` in `python/model.py`, change `ARCHITECTURE`, and **retrain from scratch**. Keep the float32 `[N,1,64,64]` input and two raw logits in O/X order. The exported model uses fixed batch size 1. Do not add a final softmax. Browser-supported operations and the 32 MiB size limit still apply. Matching parameter shapes alone cannot detect every semantic architecture change; changing the architecture version is your responsibility.

## 4. Predict and export

```sh
python python/predict.py path/to/new-photo.jpg
python python/export.py --output artifacts/model.onnx
```

Export loads the checkpoint, checks its version and class order, switches to evaluation mode, embeds weights in one file, checks ONNX, and compares eight inputs against PyTorch. A successful export validates numerical compatibility, not real-photo recognition quality.

Transfer `artifacts/model.onnx` to your phone using a method you choose, such as AirDrop, USB or your own file storage. Save it somewhere your phone’s file picker can find. The website itself does not transfer or upload it.

## 5. Use Model Playground

1. Open [the live Model Playground](https://supertweety.github.io/model-playground/).
2. Choose your local `.onnx` file and wait for **Ready**.
3. Draw an O or X, choose a photo, or use **Take photo**. The phone/browser decides how its camera picker opens.
4. Inspect **What your model sees**. This 64×64 grayscale image is the actual input.
5. Select **Run model**. Read the predicted class and both softmax scores.

**Clear image** resets the input; **Draw** starts a fresh canvas. **Remove** unloads the model. Changing either image or model clears stale predictions. Refreshing clears all selected files. You will need to select them again.

For HEIC issues in a browser, convert/export the photo as JPEG or PNG. Python HEIC support does not mean your browser can decode it. For model errors, confirm names `image` and `logits`, fixed shapes, float32, embedded weights, opset 18, and supported operators. A model that works in desktop Python may still use an operation unavailable in the browser.

High scores are not guarantees. This is a two-class model: it assigns O/X scores even to a cat, blank paper, or an entire board. Finger drawings may differ substantially from the photos used for training. Always inspect errors and report which data your evaluation covers.
