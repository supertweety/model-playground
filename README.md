# Model Playground

A shared, static website for students to test their own PyTorch models on **one handwritten O or X**. Models, drawings, and photos stay in browser memory. No accounts, uploads, analytics, backend, or student submissions.

A small **synthetic-font demo** is included: tap **Try font demo** on the [live site](https://supertweety.github.io/model-playground/), then try the printed O/X samples. It is trained only on rendered letters, not handwriting or camera photos. No labeled real-photo dataset is included. Automated checks create explicitly **UNTRAINED** fixtures under ignored `artifacts/`; these are never published.

## Open the playground

**[Launch Model Playground](https://supertweety.github.io/model-playground/)** on your phone or computer. No installation, repository download, or local server is needed to use the website.

1. Tap **Try font demo** for a quick first run, or **Choose model** to select your own `.onnx` file.
2. Draw a symbol, choose/take a photo, or use a printed sample.
3. Inspect the processed input and select **Run model**.

Student workflow: train in Python → export an ONNX file → open the live playground → select that file. Python runs in your course notebook, Colab, or your own computer; the website is already hosted for everyone.

Refreshing clears the selection, but your saved file stays in Files/Downloads. See the [phone guide](docs/phone-guide.md) for transferring and finding it. JPEG/PNG is the reliable browser photo fallback.

## Export your own network

```python
from python.playground_export import export_for_playground
path = export_for_playground(model, "artifacts/my-ox-v1.onnx")
```

Start with the [export tutorial](docs/export-tutorial.md) and [phone transfer guide](docs/phone-guide.md). The website also has a short [export & phone walkthrough](https://supertweety.github.io/model-playground/guide.html).

## Train and export

For the course example, follow the [Python training setup](docs/student-guide.md#2-prepare-your-python-training-environment-and-data) first. This environment is for training/exporting your network; students keep using the live website to test it.

```sh
python python/setup_data.py
# Add your own photos to data/{train,validation,test}/{O,X}/.
python python/train.py --epochs 20
python python/predict.py path/to/new-photo.jpg
python python/export.py --output artifacts/model.onnx
```

Read the [student guide](docs/student-guide.md) before splitting photos, and the [teacher guide](docs/teacher-guide.md) for deployment, verification, and classroom setup.

## Contract

- Self-contained ONNX, embedded weights, maximum **32 MiB = 33,554,432 bytes**.
- One float32 `image` input with fixed shape `[1,1,64,64]`.
- One float32 `logits` output with fixed shape `[1,2]`, ordered **O, X**.
- Raw logits; the website applies stable softmax. Export example uses opset 18.
- Browser-supported operations; network architecture is otherwise up to the student.
- Preprocessing version: `ox-gray64-v1`. See [the exact contract](docs/preprocessing.md).

The site validates metadata and runs trial inference. It cannot verify semantic class order, training quality, or whether a photo contains a symbol.

## Files

| Directory | Purpose |
| --- | --- |
| `website/` | Static HTML, CSS, JavaScript; the only deployed directory |
| `python/` | Training, export, prediction, preprocessing and dependencies |
| `scripts/` | Integrity-checked runtime download |
| `tests/` | Python/ONNX/browser parity and interaction checks |
| `docs/` | Student and teacher guides, contract, verification record |
| `.github/workflows/` | CI verification and GitHub Pages deployment |

ONNX Runtime Web **1.22.0** is downloaded with a pinned SHA-512 checksum. Its matching license and third-party notices are fetched from the official release tag with pinned SHA-256 checksums. All runtime resources are served locally with the site. No inference-time CDN is used. Downloaded assets, environments, datasets, checkpoints, exported models and test artifacts are ignored by Git.

## Optional: develop or run the website locally

Only needed if you want to modify the website or work on its internals. Follow the [local website setup](docs/teacher-guide.md#optional-run-the-website-locally) in the teacher/developer guide.

## Verification (maintainers)

```sh
python -m pip install -r tests/requirements.txt
python -m playwright install chromium webkit
python scripts/fetch_runtime.py
python tests/export_helper.py
python tests/verify.py
python tests/training_smoke.py
```

Linux browser setup may require `python -m playwright install --with-deps chromium webkit`.
The integration suite serves the website under `/model-playground/`, compares exact pixel-array preprocessing, PyTorch/ONNX and browser logits, exercises the UI and captures desktop/phone-emulation screenshots. See [verification status](docs/verification.md) for results and remaining real-device checks.

## Deployment (maintainers)

The intended public repository is `model-playground`. The workflow publishes only `website/` to GitHub Pages after verification. In GitHub **Settings → Pages → Build and deployment**, select **GitHub Actions**. See the [teacher guide](docs/teacher-guide.md) for the complete safe publication procedure.

This first exercise does not detect boards, blank cells, multiple symbols or arbitrary class lists. High class scores do not guarantee correctness; unrelated inputs still receive O/X scores.

## Font demo

The repository contains only the prebuilt `website/demo/basic-fonts.onnx` starter and its evaluation report. The private script and training source used to create that artifact are deliberately kept outside this public repository. See [demo details](docs/font-demo.md) for the model's scope and limitations. Only this explicitly published demo model is exempt from the model-file Git ignore rule.
