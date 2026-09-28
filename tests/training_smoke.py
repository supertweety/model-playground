"""Synthetic plumbing-only test. These images/scores are NOT a real dataset evaluation."""
from pathlib import Path
import subprocess
import sys
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts' / 'training-smoke'


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    data = ART / 'data'
    run('python/setup_data.py', '--data', data)
    for split in ['train', 'validation', 'test']:
        for label in ['O', 'X']:
            for i in range(4):
                image = Image.new('RGB', (85 + i, 91), 'white'); draw = ImageDraw.Draw(image)
                if label == 'O':
                    draw.ellipse((15 + i, 15, 65, 75), outline='black', width=7)
                else:
                    draw.line((15 + i, 15, 65, 75), fill='black', width=7)
                    draw.line((65, 15, 15 + i, 75), fill='black', width=7)
                image.save(data / split / label / f'synthetic-{i}.png')
    checkpoint = ART / 'best.pt'
    run('python/train.py', '--data', data, '--checkpoint', checkpoint, '--epochs', 2, '--batch-size', 4, '--device', 'cpu')
    saved = torch.load(checkpoint, weights_only=True)
    assert saved['epoch'] in [1, 2] and 0 <= saved['validation_accuracy'] <= 1
    assert saved['classes'] == ['O', 'X'] and saved['contract_version'] == 'ox-gray64-v1'
    run('python/predict.py', data / 'test' / 'O' / 'synthetic-0.png', '--checkpoint', checkpoint)
    run('python/export.py', '--checkpoint', checkpoint, '--output', ART / 'SYNTHETIC-ONLY.onnx')
    print('PASS: training/checkpoint/test/confusion matrix/prediction/export plumbing; no real accuracy claim.')


if __name__ == '__main__':
    main()
