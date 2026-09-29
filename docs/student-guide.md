# Student quick start

Open the [live Model Playground](https://supertweety.github.io/model-playground/) on your phone or computer. Nothing needs to be installed to use the website. Tap **Try font demo** for a first run, or download the example `.onnx` to practice selecting a file.

## 1. Bring your own PyTorch model

Use the network you designed and trained in class. This repository provides an export helper and a browser app, not a network architecture or training lessons.

Your model must accept float32 `image [1,1,64,64]` and return float32 `logits [1,2]` in **O, X** order. Outputs must be raw logits. The browser uses grayscale pixels normalized from black −1 to white +1; see the [exact input contract](preprocessing.md).

## 2. Export it

In your existing Python notebook or project, use the [export tutorial](export-tutorial.md) to install the helper's dependencies and make it importable. With your model already loaded:

```python
from python.playground_export import export_for_playground

path = export_for_playground(model, "my-ox-v1.onnx")
```

The helper checks the ONNX file and compares its outputs with PyTorch. A `.pt` checkpoint must be loaded into your own network first; changing its extension is not an export.

## 3. Save the file on your phone

Download the `.onnx` from your notebook or transfer it from your computer. Keep it in Files/Downloads or a folder called Model Playground. Follow the [phone guide](phone-guide.md) for iPhone, Android, AirDrop, USB, cloud storage and Colab instructions.

## 4. Test in the live app

1. Open [Model Playground](https://supertweety.github.io/model-playground/).
2. Choose your `.onnx` file and wait for **Ready**.
3. Draw an O or X, select/take a photo, or try a printed sample.
4. Inspect **What your model sees**, then select **Run model**.

Crop roughly square around one complete symbol. Resizing stretches rectangular images and can lose thin strokes. Finger drawings and camera photos may look very different to your model. JPEG/PNG is the reliable fallback when a browser cannot open HEIC.

**Clear image** resets the input; **Draw** starts a fresh canvas; **Remove** unloads the model. Refresh clears the selection, but the saved file stays in Files/Downloads. Select it again to continue.

High scores are not guarantees. A two-class model also assigns O/X scores to blank paper, unrelated objects and whole boards. The website cannot verify your semantic class order or model quality.
