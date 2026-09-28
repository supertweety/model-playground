"""Build the published synthetic-font demo. No handwriting or camera photos used."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import torch
from torch.utils.data import DataLoader, TensorDataset
if __package__:
    from .common import preprocess
    from .model import Net
    from .playground_export import export_for_playground
else:
    from common import preprocess
    from model import Net
    from playground_export import export_for_playground

FONTS = ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf', 'DejaVuSansMono.ttf']


def dataset(count, seed):
    rng = np.random.default_rng(seed)
    tensors, labels = [], []
    fonts = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf'
    for i in range(count):
        label = i % 2
        image = Image.new('RGB', (128, 128), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(fonts / FONTS[int(rng.integers(len(FONTS)))]), int(rng.integers(72, 123)))
        symbol = ['O', 'X'][label]
        bounds = draw.textbbox((0, 0), symbol, font=font)
        x = (128 - bounds[2] + bounds[0]) / 2 - bounds[0] + int(rng.integers(-12, 13))
        y = (128 - bounds[3] + bounds[1]) / 2 - bounds[1] + int(rng.integers(-12, 13))
        shade = int(rng.integers(0, 85))
        draw.text((x, y), symbol, font=font, fill=(shade, shade, shade))
        image = image.rotate(float(rng.uniform(-12, 12)), resample=Image.Resampling.BICUBIC, fillcolor='white')
        tensors.append(preprocess(image)); labels.append(label)
    return TensorDataset(torch.stack(tensors), torch.tensor(labels))


def accuracy(model, data):
    model.eval()
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    with torch.no_grad():
        for x, y in DataLoader(data, batch_size=64):
            for truth, prediction in zip(y, model(x).argmax(1)):
                matrix[truth, prediction] += 1
    return matrix.diag().sum().item() / len(data), matrix.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('website/demo/basic-fonts.onnx'))
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(2026)
    train, validation, test = dataset(1600, 10), dataset(320, 20), dataset(400, 30)
    model = Net(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best, selected = -1., None
    for epoch in range(8):
        model.train()
        for x, y in DataLoader(train, batch_size=64, shuffle=True):
            optimizer.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward(); optimizer.step()
        score, _ = accuracy(model, validation)
        print(f'Font demo epoch {epoch + 1}: validation {score:.1%}', flush=True)
        if score > best:
            best = score; selected = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(selected); model.eval()
    score, matrix = accuracy(model, test)
    if score < .95:
        raise RuntimeError(f'Synthetic-font test accuracy too low to ship: {score:.1%}')
    export_for_playground(model, args.output, sample_inputs=[test[i][0].unsqueeze(0) for i in range(8)])
    report = {'description': 'Teaching demo trained ONLY on synthetic printed uppercase O/X; not a handwriting or real-photo model.',
              'fonts': FONTS, 'train_count': 1600, 'validation_count': 320, 'test_count': 400,
              'seed': 2026, 'split_seeds': [10, 20, 30], 'epochs': 8,
              'validation_accuracy': best, 'synthetic_test_accuracy': score, 'confusion_matrix_true_rows_O_X': matrix,
              'evaluation_scope': 'Independent random renders of the same font families; not held-out fonts or real photographs.',
              'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
