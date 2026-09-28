"""Download only the pinned, integrity-checked ONNX Runtime Web distribution."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

VERSION = '1.22.0'
# npm's published SHA-512 integrity, pinned after initial verification.
INTEGRITY = 'sha512-Ud/+EBo6mhuaQWt/OjaOk0iNWjXqJoeeMFr6xQEERZdIZH2OWpGzuujz7lfuOBjUa6TEE/sc4nb7Da5dNL34fg=='
DEST = Path(__file__).resolve().parents[1] / 'website' / 'vendor' / 'ort'


def main():
    url = f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz'
    with urlopen(url, timeout=120) as response:
        archive = response.read()
    actual = 'sha512-' + base64.b64encode(hashlib.sha512(archive).digest()).decode()
    if actual != INTEGRITY:
        raise RuntimeError(f'Runtime integrity mismatch: {actual}')
    DEST.mkdir(parents=True, exist_ok=True)
    names = ['dist/ort.wasm.min.mjs', 'dist/ort-wasm-simd-threaded.mjs',
             'dist/ort-wasm-simd-threaded.wasm']
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        for name in names:
            member = tar.getmember('package/' + name)
            if not member.isfile():
                raise RuntimeError(f'Unexpected archive member: {name}')
            (DEST / Path(name).name).write_bytes(tar.extractfile(member).read())
    notices = {'LICENSE': '2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c', 'ThirdPartyNotices.txt': 'e9e90971a8e75a9a8ac0c6412e29c1202d079998389915aa485f46c816c3b4cc'}
    for name, digest in notices.items():
        with urlopen(f'https://raw.githubusercontent.com/microsoft/onnxruntime/v{VERSION}/{name}', timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f'License integrity mismatch: {name}')
        (DEST / name).write_bytes(data)
    (DEST / 'version.json').write_text(json.dumps({'version': VERSION, 'integrity': INTEGRITY}, indent=2) + '\n')
    print(f'Prepared ONNX Runtime Web {VERSION} in {DEST}')


if __name__ == '__main__':
    main()
