"""Train from scratch; select on validation; evaluate test once after selection."""
import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomAffine, InterpolationMode
from common import CLASSES, CONTRACT_VERSION, EXTENSIONS, load_image, preprocess, device, load_checkpoint
from model import Net


class Symbols(Dataset):
    def __init__(self, root, augment=False):
        self.samples = []
        for index, label in enumerate(CLASSES):
            folder = root / label
            images = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
            if not images:
                raise ValueError(f'No supported images in {folder}. Add JPEG, PNG, or HEIC photos.')
            self.samples.extend((p, index) for p in images)
        self.augment = RandomAffine(degrees=12, translate=(.06, .06), scale=(.9, 1.1),
                                    interpolation=InterpolationMode.NEAREST, fill=255) if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = load_image(path)
        if self.augment:
            # White composite before RGB augmentation; preprocessing itself remains shared.
            from PIL import Image
            background = Image.new('RGBA', image.size, 'white')
            image = self.augment(Image.alpha_composite(background, image).convert('RGB'))
        return preprocess(image), label


def evaluate(model, loader, target):
    model.eval()
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(target)).argmax(1).cpu()
            for truth, prediction in zip(labels, predictions):
                matrix[truth, prediction] += 1
    return matrix.diag().sum().item() / matrix.sum().item(), matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/best.pt'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    target = device() if args.device == 'auto' else torch.device('cpu')
    train = DataLoader(Symbols(args.data / 'train', True), batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(Symbols(args.data / 'validation'), batch_size=args.batch_size)
    test = DataLoader(Symbols(args.data / 'test'), batch_size=args.batch_size)
    model = Net().to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    best = -1.
    print(f'Device: {target}; class order: {CLASSES}')
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total_loss = 0.
        for images, labels in train:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images.to(target)), labels.to(target))
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
        accuracy, _ = evaluate(model, validation, target)
        print(f'Epoch {epoch + 1:02d}: loss={total_loss / len(train.dataset):.4f}, validation={accuracy:.3%}')
        if accuracy > best:
            best = accuracy
            torch.save({'state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        'contract_version': CONTRACT_VERSION, 'classes': CLASSES,
                        'architecture': model.ARCHITECTURE, 'validation_accuracy': best,
                        'epoch': epoch + 1, 'seed': args.seed}, args.checkpoint)
    load_checkpoint(args.checkpoint, model)
    accuracy, matrix = evaluate(model, test, target)
    print(f'Best validation: {best:.3%}; final test accuracy: {accuracy:.3%}')
    print('Confusion matrix: rows=true O,X; columns=predicted O,X')
    print(matrix.numpy())
    print('Reserve this test result for final reporting, not further tuning.')


if __name__ == '__main__':
    main()
