"""Check custom-architecture export and failure behavior, without a course checkpoint."""
import sys
from pathlib import Path
import torch
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python.playground_export import export_for_playground

out = ROOT / 'artifacts' / 'export-helper' / 'custom.onnx'
torch.manual_seed(10)
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(.2), torch.nn.Linear(4096, 2)).train()
original = {k: v.clone() for k, v in model.state_dict().items()}
path = export_for_playground(model, out, sample_inputs=[torch.zeros(1, 1, 64, 64)])
assert path == out and model.training and model[1].training
assert all(torch.equal(original[k], v) for k, v in model.state_dict().items())
result = ort.InferenceSession(str(out)).run(None, {'image': np.zeros((1, 1, 64, 64), np.float32)})[0]
assert result.shape == (1, 2)
previous = out.read_bytes()
for kwargs in [{'class_order': ['X', 'O']}, {'sample_inputs': [torch.zeros(64, 64)]}]:
    try:
        export_for_playground(model, out, **kwargs)
        raise AssertionError('Invalid export accepted')
    except ValueError:
        assert out.read_bytes() == previous
bad = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4096, 3))
try:
    export_for_playground(bad, out)
    raise AssertionError('Three-class model accepted')
except ValueError:
    assert out.read_bytes() == previous
print('PASS: generic architecture export, original training state/weights preserved, invalid exports preserve previous file')
