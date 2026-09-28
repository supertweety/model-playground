# Get your model onto your phone

**There is no model upload to Model Playground.** “Choose model” reads a file on your device. If you use cloud storage to transfer it, that transfer is handled by your chosen provider; the playground does not receive it.

## First try: no computer required

Open [Model Playground](https://supertweety.github.io/model-playground/) and tap **Try font demo**. It loads a small public starter model and tests a printed O. Use the printed O/X buttons and **Run model** to try other letters and font families. This demo was trained only on synthetic printed letters; it is not a handwriting or phone-photo benchmark.

To rehearse the same file-picker flow as your own model, tap **Download demo .onnx**, save it, then tap **Choose model** and select `basic-fonts.onnx`.

## iPhone / iPad

1. Export your `.onnx` file. From a Mac, AirDrop it to your phone and choose **Save to Files** if offered. Alternatively, put it in your own iCloud Drive folder so you can access it from Files.
2. For a file downloaded in Safari, look in the **Files** app → **Browse** → **Downloads**, often under **iCloud Drive**; the location depends on your download settings. Safari’s downloads button also helps locate a recent download.
3. For easy reuse, make a `Model Playground` folder in **On My iPhone/iPad** or **iCloud Drive**. Save models there with names such as `my-ox-v1.onnx`. Use Files, not Photos.
4. Return to the website, tap **Choose model**, browse to the saved file, and wait for **Ready**. If a cloud file has not downloaded yet, make it available locally first.

Apple’s [guide to finding downloads](https://support.apple.com/en-au/102440) explains the Files/Downloads locations. Picker wording can vary by OS version.

## Android

1. Download the `.onnx` file in Chrome, copy it over USB, or download a copy from your own Google Drive/OneDrive/other storage app. A cloud preview or sharing link is not the model file.
2. Find it in **Files / My Files → Downloads**. Chrome also has a **Downloads** list. You can keep models in Downloads or move them into a `Model Playground` folder.
3. Return to the website, tap **Choose model**, choose the downloaded `.onnx` file, and wait for **Ready**.

See Google’s [Chrome download instructions](https://support.google.com/chrome/answer/95759?co=GENIE.Platform%3DAndroid&hl=en). App and picker names vary by device.

## Which transfer should I choose?

| Where the model is | Simple route |
| --- | --- |
| Mac → iPhone/iPad | AirDrop → Save to Files, or your own iCloud Drive folder |
| Windows/Linux → Android | USB copy to Downloads, or your own cloud storage → download a local copy |
| Any computer → any phone | Put the file in your own private cloud folder, open its app on the phone, and save/download the `.onnx` |
| Colab/Jupyter on the phone | Export, download the file from the notebook, then select it from Downloads/Files |

Only share a cloud link if you intend others to access that model. The playground needs no public link, account, or upload permission. Keep a backup of the original `.onnx` and Python checkpoint on your computer/private storage.

## If you cannot select the file

- Verify the filename ends in **`.onnx`**, not `.pt`, `.zip`, `.html`, or `.onnx.txt`. Renaming a checkpoint does not convert it.
- Download the actual file rather than saving a cloud preview or GitHub page. Use **Download demo .onnx** for the starter.
- Extract a ZIP before selecting a model. The website expects one file with weights embedded, up to 32 MiB.
- Save a local copy if the cloud provider is missing from the picker. Try the regular Safari/Chrome browser if a chat app’s embedded browser limits downloads.
- Refresh clears the selected model from website memory, **not the saved file in Files/Downloads**. Choose that same file again. Student models are never saved by the site.
- Browser HEIC support varies; JPEG/PNG is the reliable photo fallback.

Native phone camera/file-picker behavior has not been tested on real hardware; automated checks use desktop browsers and phone emulation.
