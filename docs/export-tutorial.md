# Your PyTorch model → Model Playground

The file format is **ONNX** (`.onnx`). A `.pt` checkpoint cannot be selected directly on the website; renaming it does not convert it.

After exporting, select your file in the [live Model Playground](https://supertweety.github.io/model-playground/). Python is needed for this export step; running the website locally is not required.

## Fastest route: the course example

From the repository root, with your [Python training environment](student-guide.md#2-prepare-your-python-training-environment-and-data) activated:

```sh
python python/export.py --checkpoint checkpoints/best.pt --output artifacts/my-ox-v1.onnx
```

This loads the network from `python/model.py`. For your own model already in memory, use the function below instead.

## 1. Check your model's promise

You can use your own architecture. Its forward method must accept a float32 tensor `[1, 1, 64, 64]` and return one float32 tensor `[1, 2]` of **raw logits**, ordered **O, X**. During training, label O as 0 and X as 1. Do not put softmax in the last layer. Use the [shared image preprocessing](preprocessing.md): grayscale, black −1, white +1.

Export does not train a model or fix its class labels. If your output order is X,O, correct the label mapping or explicitly reorder those two outputs in a wrapper before exporting; simply declaring O,X is not enough.

## 2. Export a model from your own script or notebook

Run your notebook from the repository root (the folder containing `python/`). Install `python/requirements.txt` in its environment first. After your existing training code has produced `model`:

```python
from python.playground_export import export_for_playground

onnx_path = export_for_playground(
    model,                              # your trained torch.nn.Module
    "artifacts/my-ox-v1.onnx",
    class_order=("O", "X"),              # must match your training labels
)
print(onnx_path.resolve())
```

The helper is a standalone module: you may also copy `python/playground_export.py` into your own project and use `from playground_export import export_for_playground` after installing the same dependencies.

If your model is saved as a state dictionary, instantiate **your architecture** first, load its weights, and then export:

```python
import torch

# MyNetwork is the class you used during training.
model = MyNetwork()
state = torch.load("my-weights.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)  # For a wrapped checkpoint, select its state_dict key.
onnx_path = export_for_playground(model, "artifacts/my-ox-v1.onnx")
```

Keep your `.pt` checkpoint for future Python work; the `.onnx` file is the separate browser artifact. Give each experiment a clear filename such as `my-ox-v2.onnx`.

## 3. Optionally check your own images too

Eight deterministic tensors are always tested. You can add real preprocessed examples:

```python
from python.common import load_image, preprocess

sample = preprocess(load_image("my-test-symbol.jpg")).unsqueeze(0)
onnx_path = export_for_playground(
    model,
    "artifacts/my-ox-v1.onnx",
    sample_inputs=[sample],
)
```

Each sample must already be float32 `[1,1,64,64]`, normalized to [−1,1]. Successful parity means the exported calculation matches PyTorch; it does not establish classification accuracy.

## What the function does

It copies your model, moves the copy to CPU and evaluation mode, exports embedded weights with opset 18 and the correct input/output names, checks ONNX and the 32 MiB limit, and compares outputs against PyTorch. Your original model's device and training/evaluation mode are preserved. It only replaces the destination file after checks succeed. It returns a `pathlib.Path` and prints a numerical-check summary.

It cannot verify semantic class order or detect a final softmax in every possible architecture. Browser compatibility still depends on operations supported by ONNX Runtime Web. Always try the exported file on the website.

## Google Colab or another hosted notebook

If your notebook does not already have this repository, run these cells once:

```python
!git clone https://github.com/supertweety/model-playground.git
%pip install -r model-playground/python/requirements.txt
```

Restart the notebook runtime if the installer says loaded packages changed. Re-run your training/model-loading cells after restarting. Then make the helper importable without changing your working directory:

```python
import sys
sys.path.insert(0, "model-playground")
from python.playground_export import export_for_playground

onnx_path = export_for_playground(model, "my-ox-v1.onnx")
```

In Colab, download the actual file:

```python
from google.colab import files
files.download(str(onnx_path))
```

For JupyterLab, find the exported `.onnx` file in the left file browser and download it. A path inside a remote notebook is not yet a file on your phone. Save the download or transfer it using the [phone guide](phone-guide.md).

## Troubleshooting

- **`No module named python`**: open the notebook at the repository root, add that directory to `sys.path`, or use the standalone helper module.
- **Wrong shape**: add the missing batch/channel dimensions; the browser always sends `[1,1,64,64]`.
- **Checkpoint dictionary error**: pass the instantiated network after `load_state_dict`, not the dictionary or filename.
- **Nonfinite output**: inspect training, inputs, and weights for NaN/Infinity.
- **Export/unsupported operator error**: simplify that layer or use another browser-supported implementation, then retrain when the architecture changes.
- **Everything appears reversed**: confirm O=0, X=1 in your labels and output columns. The website cannot infer the intended meaning.
