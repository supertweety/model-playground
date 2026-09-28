import argparse
from pathlib import Path
import torch
from common import CLASSES, load_image, preprocess, load_checkpoint, device
from model import Net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict one complete O or X from a new photo.')
    parser.add_argument('image', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/best.pt'))
    args = parser.parse_args()
    model = Net(); load_checkpoint(args.checkpoint, model)
    target = device(); model.to(target).eval()
    with torch.no_grad():
        scores = model(preprocess(load_image(args.image)).unsqueeze(0).to(target)).softmax(1)[0].cpu()
    print(f'Prediction: {CLASSES[int(scores.argmax())]}')
    for label, score in zip(CLASSES, scores):
        print(f'{label}: {float(score):.6f}')
