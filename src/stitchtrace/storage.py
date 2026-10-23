"""POSIX single-writer transactions; committed bundles are immutable."""
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile
import stat
import zipfile
import zlib
import fcntl
import numpy as np
from .errors import require, StitchTraceError

# Explicit exceptions raised by NumPy/ZIP while parsing local array archives.
ARRAY_PARSE_ERRORS = (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile,
                      zlib.error, RuntimeError, NotImplementedError)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path, chunk=1 << 22) -> str:
    """Streamed SHA-256 so verification never holds a whole file in memory."""
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(chunk):
            h.update(block)
    return h.hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'DUPLICATE_KEY', str(key))
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=unique)
    except (ValueError, UnicodeError) as e:
        raise StitchTraceError('INVALID_JSON', str(path)) from e


def safe_path(root, name):
    require(isinstance(name, str), 'UNSAFE_PATH', str(name))
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and not p.is_absolute() and '\\' not in name
            and all(s not in ('', '.', '..') for s in name.split('/')), 'UNSAFE_PATH', str(name))
    target = Path(root).joinpath(*p.parts)
    for part in [target, *target.parents]:
        require(not part.is_symlink(), 'SYMLINK', str(part))
    return target


def inventory(root):
    """Reject links and special entries; only the root manifest is excluded."""
    def unreadable(error):
        raise StitchTraceError('INVENTORY_UNREADABLE', str(error)) from error

    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=unreadable):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            require(not stat.S_ISLNK(mode), 'SYMLINK', str(path))
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'NONREGULAR_FILE', str(path))
            relative = path.relative_to(root).as_posix()
            if stat.S_ISREG(mode) and relative != 'manifest.json':
                result[relative] = path
    return result


def atomic_bytes(path, data):
    path = Path(path)
    safe_path(path.parent, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_array(path, array):
    import io
    out = io.BytesIO()
    np.save(out, np.asarray(array), allow_pickle=False)
    atomic_bytes(path, out.getvalue())


@contextmanager
def transaction(destination, contract):
    """Hold OS lock throughout capture. A crash preserves staging for exact resume."""
    destination = Path(destination).absolute()
    safe_path(destination.parent, destination.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = safe_path(destination.parent, '.' + destination.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if destination.exists():
            from .bundle import Bundle
            bundle = Bundle.open(destination)
            require(bundle.meta['contract'] == contract, 'CONTRACT_CONFLICT', str(destination))
            yield None
            return
        staging = safe_path(destination.parent, '.' + destination.name + '.work')
        staging.mkdir(exist_ok=True)
        identity = staging / 'transaction.json'
        if identity.exists():
            require(read_json(identity)['contract'] == contract, 'CONTRACT_CONFLICT', str(staging))
        else:
            atomic_bytes(identity, encoded({'contract': contract, 'status': 'pending'}))
        try:
            yield staging
            require((staging / 'manifest.json').is_file(), 'INCOMPLETE', 'manifest absent')
            from .bundle import Bundle
            Bundle.open(staging)
            os.rename(staging, destination)
            parent = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        except BaseException as e:
            history = staging / 'failures'
            history.mkdir(exist_ok=True)
            import uuid
            atomic_bytes(history / (uuid.uuid4().hex + '.json'), encoded({'type': type(e).__name__, 'error': str(e)}))
            raise


def cache_array(root, key, factory):
    """One atomic NPZ cache record (array + digest); corruption never recomputed silently."""
    import io
    token = digest(encoded(key))
    path = safe_path(root, 'cache/' + token + '.npz')
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as file:
                value = file['value']
                expected = str(file['sha'].item())
            require(digest(value.tobytes()) == expected, 'CACHE_CORRUPT', token)
            return value, True
        except ARRAY_PARSE_ERRORS as e:
            raise StitchTraceError('CACHE_CORRUPT', token) from e
    value = np.asarray(factory())
    require(not value.dtype.hasobject, 'INVALID_ARRAY', token)
    stream = io.BytesIO()
    np.savez_compressed(stream, value=value, sha=np.array(digest(value.tobytes())))
    atomic_bytes(path, stream.getvalue())
    return value, False


def commit(root, meta):
    atomic_bytes(root / 'bundle.json', encoded(meta))
    files = {}
    for name, p in sorted(inventory(root).items()):
        files[name] = {'sha256': file_digest(p), 'bytes': p.stat().st_size}
    atomic_bytes(root / 'manifest.json', encoded({'schema': 1, 'files': files}))
