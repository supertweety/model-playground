// Small bounded protobuf reader for ONNX's declared I/O contract. Inference is
// still validated by ONNX Runtime. No model-supplied path is ever fetched.
function fields(bytes) {
  let pos = 0;
  const result = [];
  function integer() {
    let n = 0, scale = 1;
    for (let i = 0; i < 10; i++) {
      if (pos >= bytes.length) throw new Error('Truncated ONNX file.');
      const b = bytes[pos++]; n += (b & 127) * scale;
      if (!(b & 128)) return n;
      scale *= 128;
    }
    throw new Error('Invalid ONNX integer.');
  }
  while (pos < bytes.length) {
    const tag = integer(), wire = tag % 8, id = Math.floor(tag / 8);
    if (!id) throw new Error('Invalid ONNX field.');
    let value;
    if (wire === 0) value = integer();
    else if (wire === 2) {
      const length = integer();
      if (!Number.isSafeInteger(length) || pos + length > bytes.length) throw new Error('Truncated ONNX data.');
      value = bytes.subarray(pos, pos + length); pos += length;
    } else if (wire === 1 || wire === 5) {
      const length = wire === 1 ? 8 : 4;
      if (pos + length > bytes.length) throw new Error('Truncated ONNX data.');
      pos += length; continue;
    } else throw new Error('Unsupported ONNX encoding.');
    result.push({id, value});
  }
  return result;
}
const get = (f, id) => f.find(v => v.id === id)?.value;
const all = (f, id) => f.filter(v => v.id === id).map(v => v.value);
const message = value => {
  if (!(value instanceof Uint8Array)) throw new Error('Missing ONNX metadata.');
  return fields(value);
};
const string = value => new TextDecoder().decode(value);
function checkValue(value, name, dims) {
  const f = message(value);
  const tensor = message(get(message(get(f, 2)), 1));
  const shape = message(get(tensor, 2));
  const actual = all(shape, 1).map(d => get(message(d), 1));
  if (string(get(f, 1)) !== name || get(tensor, 1) !== 1 || JSON.stringify(actual) !== JSON.stringify(dims)) {
    throw new Error(`Expected float32 ${name} [${dims.join(', ')}], with fixed dimensions.`);
  }
}
function checkTensor(value) {
  const f = message(value);
  if (get(f, 14) === 1 || all(f, 13).length) throw new Error('External weights are not supported. Export one ONNX file with embedded weights.');
}
function checkGraph(value, depth = 0) {
  if (depth > 32) throw new Error('Model graph nesting is too deep.');
  const graph = message(value);
  all(graph, 5).forEach(checkTensor);
  for (const s of all(graph, 15)) {
    const sparse = message(s); checkTensor(get(sparse, 1)); checkTensor(get(sparse, 2));
  }
  checkNodes(all(graph, 1), depth);
  return graph;
}
function checkNodes(nodes, depth) {
  for (const node of nodes) checkAttributes(all(message(node), 5), depth);
}
function checkAttributes(attributes, depth) {
  for (const attribute of attributes) {
    const a = message(attribute);
    for (const id of [5, 10]) all(a, id).forEach(checkTensor);
    for (const id of [6, 11]) all(a, id).forEach(g => checkGraph(g, depth + 1));
    for (const id of [22, 23]) for (const s of all(a, id)) {
      const sparse = message(s); checkTensor(get(sparse, 1)); checkTensor(get(sparse, 2));
    }
  }
}
export function validateContract(buffer) {
  const model = message(new Uint8Array(buffer));
  for (const info of all(model, 20)) {
    const training = message(info);
    for (const id of [1, 2]) all(training, id).forEach(g => checkGraph(g));
  }
  for (const fn of all(model, 25)) {
    const f = message(fn);
    checkNodes(all(f, 7), 0);
    checkAttributes(all(f, 11), 0);
  }
  const graph = checkGraph(get(model, 7));
  const inputs = all(graph, 11), outputs = all(graph, 12);
  if (inputs.length !== 1 || outputs.length !== 1) throw new Error('The model must have exactly one input and one output.');
  checkValue(inputs[0], 'image', [1, 1, 64, 64]);
  checkValue(outputs[0], 'logits', [1, 2]);
}
