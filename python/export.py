"""Export a self-contained opset-18 model and compare multiple inputs."""
import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from common import load_checkpoint, CONTRACT_VERSION, CLASSES
from model import Net


def export_model(model, destination):
    model.cpu().eval()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(model, example, str(destination), input_names=['image'], output_names=['logits'],
                      opset_version=18, export_params=True, dynamo=False, external_data=False)
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    if destination.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('Export exceeds the website 32 MiB limit')
    for value, name, shape in [(graph.graph.input[0], 'image', [1, 1, 64, 64]),
                                (graph.graph.output[0], 'logits', [1, 2])]:
        tensor = value.type.tensor_type
        if value.name != name or tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != shape:
            raise ValueError(f'Invalid exported contract for {name}')
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError('Exactly one input and output required')
    for tensor in graph.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError('Export has external weights')
    onnx.helper.set_model_props(graph, {'preprocessing_contract': CONTRACT_VERSION, 'class_order': ','.join(CLASSES)})
    onnx.save_model(graph, destination, save_as_external_data=False)
    session = ort.InferenceSession(str(destination), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(42)
    inputs = [np.full((1, 1, 64, 64), fill, dtype=np.float32) for fill in [-1, 0, 1]]
    inputs += [rng.uniform(-1, 1, (1, 1, 64, 64)).astype(np.float32) for _ in range(5)]
    worst = 0.
    with torch.no_grad():
        for x in inputs:
            expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(['logits'], {'image': x})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            worst = max(worst, float(np.max(np.abs(actual - expected))))
    print(f'Exported {destination}; PyTorch/ONNX parity passed on {len(inputs)} inputs; max error {worst:.3g}')
    return worst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/best.pt'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/model.onnx'))
    args = parser.parse_args()
    model = Net(); load_checkpoint(args.checkpoint, model)
    export_model(model, args.output)
