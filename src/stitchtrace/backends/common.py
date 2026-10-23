import hashlib
import inspect
import importlib
import importlib.metadata
import json
from pathlib import Path
import threading
import numpy as np
from ..storage import encoded, digest
from ..errors import require

HOOK_LOCK = threading.RLock()


def compatible(package, allow_unverified=False):
    """Check the installed upstream against the reviewed version, source hashes and signatures.

    With ``allow_unverified`` a mismatch is recorded in the returned identity instead of raised;
    the hooked signatures must still match because capture depends on them.
    """
    lock = json.loads(Path(__file__).with_name('compatibility.json').read_text())
    installed = importlib.metadata.version(package)
    problems = []
    if installed != lock['versions'][package]:
        problems.append(f'version {installed} != reviewed {lock["versions"][package]}')
    selected = {n: s for n, s in lock['modules'].items() if n.startswith(package + '.')}
    for name, spec in selected.items():
        module = importlib.import_module(name)
        if digest(Path(module.__file__).read_bytes()) != spec['sha256']:
            problems.append(f'source {name} differs from reviewed hash')
        for attr, signature in spec['signatures'].items():
            obj = module
            for part in attr.split('.'):
                obj = getattr(obj, part)
            require(str(inspect.signature(obj)) == signature, 'SIGNATURE_UNSUPPORTED', name + '.' + attr)
    if problems and not allow_unverified:
        code = 'VERSION_UNSUPPORTED' if problems[0].startswith('version') else 'SOURCE_UNSUPPORTED'
        require(False, code, package + ': ' + '; '.join(problems))
    package_root = Path(__file__).resolve().parents[1]
    implementation = {str(p.relative_to(package_root)): digest(p.read_bytes()) for p in sorted(package_root.rglob('*.py'))}
    return {'package': package, 'version': installed, 'reviewed_version': lock['versions'][package],
            'verified': not problems, 'problems': problems, 'modules': selected,
            'implementation': {'version': _own_version(), 'files': implementation}}


def _own_version():
    try:
        return importlib.metadata.version('stitchtrace')
    except importlib.metadata.PackageNotFoundError:
        from .. import __version__
        return __version__


def contract(image, parameters, source, predictor_id):
    a = np.ascontiguousarray(image)
    require(a.ndim in (2, 3) and not a.dtype.hasobject and np.all(np.isfinite(a)), 'INVALID_IMAGE', 'finite 2D/3D input required')
    return digest(encoded({'shape': a.shape, 'dtype': a.dtype.str, 'bytes': digest(a.tobytes()),
                           'parameters': parameters, 'source': source, 'predictor': predictor_id}))


def bounds(slices):
    return [[int(s.start), int(s.stop)] for s in slices]
