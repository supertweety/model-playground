"""Export the course checkpoint, or import export_for_playground for your own model."""
import argparse
from pathlib import Path
if __package__:
    from .common import load_checkpoint
    from .model import Net
    from .playground_export import export_for_playground
else:
    from common import load_checkpoint
    from model import Net
    from playground_export import export_for_playground


def export_model(model, destination):
    """Compatibility wrapper for existing course scripts."""
    return export_for_playground(model, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/model.onnx'))
    args = parser.parse_args()
    model = Net(); load_checkpoint(args.checkpoint, model)
    export_for_playground(model, args.output)
