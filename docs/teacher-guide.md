# Teacher setup guide

## Local setup

Start from the repository README commands. The verified reference environment is Python 3.10 on Apple Silicon with torch 2.7.1 / torchvision 0.22.1. Python 3.10–3.12 is the recommended range; install wheels appropriate to the OS. `pip check` verifies resolved dependency compatibility. On Linux, CPU-only PyTorch avoids a large CUDA installation:

```sh
python -m pip install --upgrade pip==25.1.1
python -m pip install --no-deps torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r python/requirements.txt
```

The first Linux command downloads only the CPU torch/torchvision wheels; the following requirements install resolves their dependencies from PyPI. Run both commands.

The website has no frontend build step. `python scripts/fetch_runtime.py` prepares the pinned JS, WASM, license and notices under ignored `website/vendor/ort/`. The downloader validates archive and license integrity. Run it when checking out a fresh copy. The site uses single-thread WASM so GitHub Pages does not need special cross-origin isolation headers.

Serve only `website/`, never the repository root. The website works at `/` and a repository subpath such as `/model-playground/`. Files use relative paths; the WASM directory is resolved relative to the JavaScript module. On a real phone, use the HTTPS Pages URL. A local server bound to `127.0.0.1` is available only on the host computer.

## Verify before class

```sh
python -m pip install -r tests/requirements.txt
python -m playwright install chromium webkit
python scripts/fetch_runtime.py
python tests/export_helper.py
python tests/verify.py
python tests/training_smoke.py
```

The integration suite uses seeded **untrained** CNNs. Its parity checks do not establish recognition accuracy. It saves screenshots and a numerical/network report under ignored `artifacts/verification/`. The training smoke test generates synthetic plumbing data only and exercises checkpoint selection, evaluation, prediction and export. Never use its scores as assignment results.

Manually test a real iPhone/Android phone: choose an ONNX file from Files/Downloads, take a photograph through the camera picker, verify orientation, run inference, clear, replace the model, and refresh. Mobile emulation cannot establish real camera permissions, picker behavior, memory limits or device performance. Test browser HEIC support separately; JPEG/PNG is the fallback.

## Publish safely to GitHub Pages

Intended repository: **public `model-playground`**. The project does not require Sites hosting or a third-party deployment service.

1. Authenticate: `gh auth login -h github.com`, then `gh auth status`.
2. Check the signed-in account (`gh api user --jq .login`). Query `gh api repos/OWNER/model-playground`. If it exists, **stop**: do not overwrite or push this project into it. Only proceed if the authenticated API returns an explicit 404; a network or authentication failure does not prove absence.
3. From this project directory, initialize a new repository if needed:

   ```sh
   git init -b main
   git add README.md .gitignore .github website python scripts tests docs
   git status --short
   git commit -m "Build Model Playground for O/X model testing"
   gh repo create OWNER/model-playground --public --source=. --remote=origin
   git push -u origin main
   ```

   Inspect the staged files before committing: no student data/models, environments, test artifacts or downloaded runtime binaries should be present. The explicitly published `website/demo/basic-fonts.onnx` is the only model exception. An existing destination causes repository creation to fail; never work around that by force-pushing.

4. In GitHub **Settings → Pages → Build and deployment**, choose **GitHub Actions**. Alternatively, create the Pages configuration with `gh api --method POST repos/OWNER/model-playground/pages -f build_type=workflow` if Pages is not configured.
5. Run **Deploy website** manually in Actions, or push a commit to `main`. The workflow verifies the implementation, downloads the pinned runtime, and uploads **only `website/`**. Set Pages before triggering the deploy run to avoid a configuration race on the initial push.
6. Check that the deployment succeeds and open the URL displayed by the deployment. For a normal repository it is `https://OWNER.github.io/model-playground/`.

The workflow requires repository Pages support and Actions permissions `pages: write` and `id-token: write`; it uses the `github-pages` environment. Fork pull requests run checks but cannot deploy. Public repositories are readable by everyone, so keep actual student files outside version control.

## Privacy and limitations

The app reads local files into memory, uses locally served runtime files, and has no network upload calls, analytics or storage writes. A restrictive content-security policy disallows third-party resources. The verification suite records requests through model/photo selection and inference and fails on non-GET or non-local requests. Hosting providers still receive ordinary requests for public website assets, which include normal connection metadata; selected file contents are not sent.

Refresh clears state. Guaranteed offline startup is not implemented. Photos are limited to 40 MiB and 40 million decoded pixels to bound ordinary memory use; choose a smaller JPEG for very large phone photos. Model files are limited to 32 MiB, but computationally large models can still be slow or exceed device memory. The trial run cannot prove semantic class order, training quality, calibration or support for every possible input-dependent execution path.

No photos/submissions are collected, no accounts are used, and the optional demo is trained only on synthetic printed letters. This is a single-symbol O/X exercise, not a board detector or blank-cell classifier.

## Starter demo and custom-model export

Students can use **Try font demo** without a file transfer, or download its `.onnx` to practice their phone’s file picker. The five-font synthetic training source and evaluation scope are in [font-demo.md](font-demo.md). This is the only committed/published model; student models and test fixtures remain ignored. The [export tutorial](export-tutorial.md) supports arbitrary compatible `torch.nn.Module` architectures and hosted notebooks. The [phone guide](phone-guide.md) distinguishes private cloud transfer from local selection; no upload service has been added.
