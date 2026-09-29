"""Integration checks using UNTRAINED fixtures; not a model-accuracy benchmark."""
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
from pathlib import Path
import sys
from threading import Thread
import numpy as np
from PIL import Image, ImageDraw
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import torch
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from common import preprocess_rgba, load_image
from playground_export import export_for_playground


def inference_fixture():
    # Random projection solely for export/inference checks. Never trained or
    # offered as an assignment architecture; no training code is needed here.
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4096, 2)).eval()

ART = ROOT / 'artifacts' / 'verification'
ART.mkdir(parents=True, exist_ok=True)
REPORT = {'fixture': 'UNTRAINED, seeded weights. No claim of recognition accuracy.', 'checks': []}


def wait_for(page, expression, timeout=30000):
    # Poll through Playwright's evaluate API; wait_for_function internally uses
    # eval(), which this site's CSP deliberately disallows.
    deadline = time.monotonic() + timeout / 1000
    while time.monotonic() < deadline:
        if page.evaluate(expression):
            return
        time.sleep(.05)
    raise AssertionError(f'Timed out: {expression}; status: {page.locator("#model-status").inner_text()}')


def passed(name):
    REPORT['checks'].append(name)
    print('PASS:', name, flush=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'website'), **kwargs)

    def do_GET(self):
        if not self.path.startswith('/model-playground/'):
            self.send_error(404); return
        self.path = self.path[len('/model-playground'):]
        super().do_GET()

    def log_message(self, *_):
        pass


def model_file(name, shape=(1, 1, 64, 64), input_name='image', output_name='logits', dtype=TensorProto.FLOAT, external=False):
    graph = helper.make_graph([helper.make_node('Constant', [], [output_name], value=helper.make_tensor('v', TensorProto.FLOAT, [1, 2], [0., 1.]))],
                              name, [helper.make_tensor_value_info(input_name, dtype, shape)],
                              [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 2])])
    if external:
        weight = helper.make_tensor('external', TensorProto.FLOAT, [1], [0.])
        weight.ClearField('float_data'); weight.data_location = TensorProto.EXTERNAL
        entry = weight.external_data.add(); entry.key = 'location'; entry.value = 'DO-NOT-FETCH.bin'
        graph.initializer.append(weight)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)], ir_version=10)
    path = ART / (name + '.onnx'); onnx.save(model, path); return path


def main():
    demo_report = json.loads((ROOT / 'website/demo/basic-fonts.json').read_text())
    assert hashlib.sha256((ROOT / 'website/demo/basic-fonts.onnx').read_bytes()).hexdigest() == demo_report['sha256']
    torch.set_num_threads(1); torch.manual_seed(42)
    net = inference_fixture()
    fixture = ART / 'UNTRAINED.onnx'
    export_for_playground(net, fixture)
    passed('PyTorch vs ONNX on 8 inputs, atol=1e-5 rtol=1e-4')
    torch.manual_seed(73)
    second = ART / 'UNTRAINED-replacement.onnx'; export_for_playground(inference_fixture(), second)
    image = Image.new('RGB', (137, 91), 'white')
    ImageDraw.Draw(image).line((25, 15, 112, 76), fill='black', width=12)
    ImageDraw.Draw(image).line((112, 15, 25, 76), fill='black', width=12)
    image.save(ART / 'symbol.png')
    rgba = np.array(image.convert('RGBA'))
    _, normalized = preprocess_rgba(rgba)
    session = ort.InferenceSession(str(fixture), providers=['CPUExecutionProvider'])
    logits = session.run(['logits'], {'image': normalized[None]})[0][0]
    scores = np.exp(logits.astype(float) - logits.max()); scores /= scores.sum()
    REPORT['python_logits'] = logits.tolist(); REPORT['python_scores'] = scores.tolist()
    oriented = Image.new('RGB', (60, 100), 'white'); ImageDraw.Draw(oriented).rectangle((0, 0, 29, 49), fill='black')
    exif = Image.Exif(); exif[274] = 6
    oriented.save(ART / 'oriented.jpg', exif=exif, quality=100, subsampling=0)
    assert load_image(ART / 'oriented.jpg').size == (100, 60)
    oriented.save(ART / 'phone.heic', format='HEIF', quality=95)
    assert load_image(ART / 'phone.heic').size == (60, 100)
    passed('Python JPEG EXIF orientation and HEIC decoding')
    rng = np.random.default_rng(3)
    arrays = [rng.integers(0, 256, (h, w, 4), dtype=np.uint8) for h, w in [(1, 1), (3, 5), (91, 137), (128, 31), (64, 64)]]
    arrays += [np.array([[[0, 0, 0, 0], [0, 0, 0, 255], [255, 255, 255, 255], [20, 100, 233, 128]]], dtype=np.uint8)]
    (ART / 'invalid.onnx').write_bytes(b'not an onnx model')
    (ART / 'invalid.png').write_bytes(b'not an image')
    invalid_models = [ART / 'invalid.onnx', model_file('bad-shape', (1, 3, 64, 64)),
                      model_file('bad-name', input_name='wrong'), model_file('bad-type', dtype=TensorProto.DOUBLE),
                      model_file('external', external=True), model_file('dynamic', ('N', 1, 64, 64))]
    nonfinite = model_file('nonfinite')
    graph = onnx.load(nonfinite)
    graph.graph.node[0].attribute[0].t.float_data[0] = float('nan')
    onnx.save(graph, nonfinite)
    unsupported = model_file('unsupported')
    graph = onnx.load(unsupported)
    graph.graph.node[0].op_type = 'NotAnOperator'
    onnx.save(graph, unsupported)
    oversized = ART / 'oversized.onnx'
    with oversized.open('wb') as file:
        file.truncate(32 * 1024 * 1024 + 1)
    invalid_models.extend([nonfinite, unsupported, oversized])
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_port}/model-playground/'
    try:
        with sync_playwright() as p:
            for engine_name in ['chromium', 'webkit']:
                browser = getattr(p, engine_name).launch()
                context = browser.new_context(viewport={'width': 1440, 'height': 1100})
                page = context.new_page(); requests = []; errors = []
                page.on('request', lambda request: requests.append((request.url, request.method, request.post_data)))
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(base)
                assert page.locator('#run').is_disabled()
                for arr in arrays:
                    actual = page.evaluate('''async ({pixels,width,height}) => {
                      const {preprocessRGBA} = await import('./preprocess.js');
                      const r = preprocessRGBA(Uint8Array.from(pixels), width, height);
                      return {gray:Array.from(r.gray), tensor:Array.from(r.tensor)};
                    }''', {'pixels': arr.flatten().tolist(), 'width': arr.shape[1], 'height': arr.shape[0]})
                    gray, tensor = preprocess_rgba(arr)
                    np.testing.assert_array_equal(actual['gray'], gray.flatten())
                    np.testing.assert_array_equal(np.array(actual['tensor'], np.float32), tensor.flatten())
                passed(f'{engine_name}: exact Python/JS RGBA parity on 6 arrays including transparency and non-square inputs')
                extremes = page.evaluate("async () => (await import('./preprocess.js')).softmax([10000, 9999])")
                np.testing.assert_allclose(extremes, [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])
                for invalid in invalid_models:
                    page.locator('#model-file').set_input_files(invalid)
                    wait_for(page, "() => document.querySelector('#model-status').textContent.startsWith('Could not load model')")
                    assert page.locator('#run').is_disabled()
                passed(f'{engine_name}: rejects malformed, wrong shape/name/type, dynamic, external-weight, nonfinite, unsupported and oversized models')
                page.locator('#model-file').set_input_files(fixture)
                wait_for(page, "() => document.querySelector('#model-status').textContent.includes('Ready')", timeout=90000)
                page.locator('#photo-file').set_input_files(ART / 'symbol.png')
                wait_for(page, "() => document.querySelector('#image-status').textContent.includes('137 × 91')")
                processed = page.evaluate("Array.from(document.querySelector('#processed').getContext('2d').getImageData(0,0,64,64).data).filter((_,i)=>i%4===0)")
                np.testing.assert_array_equal(processed, preprocess_rgba(rgba)[0].flatten())
                page.locator('#run').click(); wait_for(page, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                displayed = [float(page.locator('#score-' + c).inner_text().replace('%', '')) / 100 for c in ['o', 'x']]
                np.testing.assert_allclose(displayed, scores, atol=.000051, rtol=0)
                # Actual runtime logits, using an in-memory File obtained from the file picker.
                # The app resets the picker, so set a detached input for the direct numerical check.
                page.evaluate("() => { const f=document.createElement('input'); f.id='test-model'; f.type='file'; f.hidden=true; document.body.append(f); }")
                page.locator('#test-model').set_input_files(fixture)
                actual = page.evaluate('''async pixels => {
                  const ort = await import('./vendor/ort/ort.wasm.min.mjs');
                  const f = document.querySelector('#test-model').files[0];
                  const s = await ort.InferenceSession.create(await f.arrayBuffer(), {executionProviders:['wasm']});
                  const t = new ort.Tensor('float32', Float32Array.from(pixels), [1,1,64,64]);
                  const result = await s.run({image:t}); const values=Array.from(result.logits.data);
                  result.logits.dispose(); t.dispose(); await s.release(); document.querySelector('#test-model').remove();
                  return values;
                }''', normalized.flatten().tolist())
                np.testing.assert_allclose(actual, logits, atol=1e-5, rtol=1e-4)
                REPORT[engine_name + '_logits'] = actual
                passed(f'{engine_name}: actual WASM inference and displayed scores match Python on deterministic PNG')
                page.screenshot(path=str(ART / f'{engine_name}-desktop.png'), full_page=True)
                page.locator('#photo-file').set_input_files(ART / 'oriented.jpg')
                wait_for(page, "() => document.querySelector('#image-status').textContent.includes('100 × 60')")
                assert page.locator('#prediction').inner_text() == '—'
                browser_gray = page.evaluate("Array.from(document.querySelector('#processed').getContext('2d').getImageData(0,0,64,64).data).filter((_,i)=>i%4===0)")
                python_gray = preprocess_rgba(np.array(load_image(ART / 'oriented.jpg')))[0].flatten()
                assert np.max(np.abs(np.array(browser_gray) - python_gray)) <= 2
                passed(f'{engine_name}: EXIF orientation matches Python (JPEG tolerance <=2 gray levels)')
                page.locator('#photo-file').set_input_files(ART / 'invalid.png')
                wait_for(page, "() => document.querySelector('#image-status').textContent.startsWith('Could not open photo')")
                assert page.locator('#run').is_disabled()
                page.locator('#photo-file').set_input_files(ART / 'symbol.png')
                wait_for(page, "() => !document.querySelector('#run').disabled")
                page.locator('#clear').click()
                assert page.locator('#prediction').inner_text() == '—' and page.locator('#run').is_disabled()
                box = page.locator('#drawing').bounding_box()
                page.mouse.move(box['x'] + 60, box['y'] + 60); page.mouse.down()
                page.mouse.move(box['x'] + 220, box['y'] + 220, steps=12); page.mouse.up()
                assert page.locator('#run').is_enabled()
                page.locator('#run').click(); wait_for(page, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                page.locator('#model-file').set_input_files(second)
                assert page.locator('#prediction').inner_text() == '—'
                wait_for(page, "() => document.querySelector('#model-status').textContent.includes('Ready')")
                page.locator('#run').click(); wait_for(page, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                page.locator('#model-file').set_input_files(ART / 'invalid.onnx')
                wait_for(page, "() => document.querySelector('#model-status').textContent.includes('Could not load')")
                assert page.locator('#prediction').inner_text() == '—'
                page.locator('#model-file').set_input_files(fixture)
                wait_for(page, "() => document.querySelector('#model-status').textContent.includes('Ready')")
                page.locator('#reset-model').click(); assert page.locator('#run').is_disabled()
                passed(f'{engine_name}: drawing, photo selection, clear, invalid-image recovery, invalid-model recovery, replacement and remove')
                page.locator('#try-demo').click()
                wait_for(page, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                for font in ['sans-serif', 'serif', 'monospace']:
                    page.locator('#sample-font').select_option(font)
                    for letter in ['O', 'X']:
                        page.locator('#sample-' + letter.lower()).click()
                        assert page.locator('#prediction').inner_text() == '—'
                        page.locator('#run').click()
                        wait_for(page, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                        assert page.locator('#prediction').inner_text() == letter, (engine_name, font, letter)
                with page.expect_download() as download:
                    page.locator('a[download="basic-fonts.onnx"]').click()
                downloaded = ART / f'{engine_name}-download.onnx'
                download.value.save_as(downloaded)
                assert downloaded.read_bytes() == (ROOT / 'website/demo/basic-fonts.onnx').read_bytes()
                page.locator('#model-file').set_input_files(downloaded)
                wait_for(page, "() => document.querySelector('#model-status').textContent.includes('Ready')")
                passed(f'{engine_name}: demo inference on O/X in three browser font families, ONNX download and re-selection')
                page.screenshot(path=str(ART / f'{engine_name}-demo-desktop.png'), full_page=True)
                page.reload(); assert page.locator('#run').is_disabled()
                assert page.locator('#drawing-hint').is_visible()
                assert page.evaluate('localStorage.length + sessionStorage.length') == 0
                assert not errors, errors
                assert all(url.startswith(base) and method == 'GET' and body is None for url, method, body in requests), requests
                assert not any('DO-NOT-FETCH' in url for url, _, _ in requests)
                REPORT[engine_name + '_requests'] = sorted(set(url.replace(base, '') for url, _, _ in requests))
                passed(f'{engine_name}: refresh resets state; all observed requests are same-origin static GETs, no uploads/storage')
                mobile = browser.new_context(viewport={'width':390,'height':844}, device_scale_factor=2, is_mobile=True, has_touch=True)
                phone = mobile.new_page(); phone.goto(base)
                assert phone.evaluate('document.documentElement.scrollWidth <= innerWidth')
                phone.locator('#model-file').set_input_files(fixture)
                wait_for(phone, "() => document.querySelector('#model-status').textContent.includes('Ready')", timeout=90000)
                phone.locator('#drawing').scroll_into_view_if_needed()
                b = phone.locator('#drawing').bounding_box()
                phone.touchscreen.tap(b['x'] + b['width']/2, b['y'] + b['height']/2)
                assert phone.locator('#run').is_enabled()
                phone.locator('#run').click(); wait_for(phone, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                phone.screenshot(path=str(ART / f'{engine_name}-mobile.png'), full_page=True)
                assert phone.locator('#camera-file').get_attribute('capture') == 'environment'
                phone.locator('#camera-file').set_input_files(ART / 'symbol.png')
                wait_for(phone, "() => document.querySelector('#image-status').textContent.includes('137 × 91')")
                phone.locator('#clear').click(); assert phone.locator('#run').is_disabled()
                phone.locator('#try-demo').click()
                wait_for(phone, "() => document.querySelector('#run-status').textContent.includes('Complete')")
                assert phone.locator('#prediction').inner_text() == 'O'
                phone.screenshot(path=str(ART / f'{engine_name}-demo-mobile.png'), full_page=True)
                phone.set_viewport_size({'width':320,'height':740})
                assert phone.evaluate('document.documentElement.scrollWidth <= innerWidth')
                phone.evaluate("document.documentElement.style.fontSize='200%'")
                # At enlarged text, content should reflow with no page-level horizontal scroll.
                assert phone.evaluate('document.documentElement.scrollWidth <= innerWidth')
                passed(f'{engine_name}: 390/320px phone emulation, touch, camera input selection, and 200% text reflow')
                phone.goto(base + 'guide.html')
                assert phone.locator('h1').inner_text() == 'Your model, ready to try.'
                assert phone.evaluate('document.documentElement.scrollWidth <= innerWidth')
                phone.screenshot(path=str(ART / f'{engine_name}-guide-mobile.png'), full_page=True)
                browser.close()
    finally:
        server.shutdown()
    REPORT['unverified'] = ['Real phone hardware camera/file picker', 'Real-photo recognition accuracy; the demo uses synthetic fonts only']
    (ART / 'report.json').write_text(json.dumps(REPORT, indent=2) + '\n')
    print(f'Report: {ART / "report.json"}')


if __name__ == '__main__':
    main()
