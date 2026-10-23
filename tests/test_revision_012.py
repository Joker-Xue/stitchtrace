"""Independent-review R1/R2 and neighboring parser/traversal boundaries."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest
from stitchtrace import Bundle, StitchTraceError
from stitchtrace.storage import commit, inventory
from test_audit_regressions import valid


def cli_reject(root, code):
    result = subprocess.run([sys.executable, '-m', 'stitchtrace.cli', 'verify', str(root)],
                            capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stderr)['error'] == code
    assert 'Traceback' not in result.stderr


def test_unreadable_directory_api_cli(tmp_path):
    root = tmp_path/'bundle'
    valid(root)
    hidden = root/'hidden'
    hidden.mkdir()
    (hidden/'extra.json').write_text('{}')
    hidden.chmod(0)
    try:
        try:
            list(hidden.iterdir())
        except PermissionError:
            pass
        else:
            pytest.skip('host privilege bypasses directory permissions; injected traversal test remains active')
        with pytest.raises(StitchTraceError, match='INVENTORY_UNREADABLE'):
            Bundle.open(root)
        cli_reject(root, 'INVENTORY_UNREADABLE')
    finally:
        hidden.chmod(0o700)


def test_legal_empty_directory(tmp_path):
    root = tmp_path/'bundle'
    valid(root)
    (root/'empty').mkdir()
    Bundle.open(root)


@pytest.mark.parametrize('error',[PermissionError(13,'denied'), OSError(5,'I/O failure')])
def test_walk_error_propagates(tmp_path, monkeypatch, error):
    def failed_walk(root, *, followlinks, onerror=None):
        if onerror:
            onerror(error)
        return iter(())
    monkeypatch.setattr(os, 'walk', failed_walk)
    with pytest.raises(StitchTraceError, match='INVENTORY_UNREADABLE'):
        inventory(tmp_path)


def malformed_archive(path, kind):
    stream = io.BytesIO()
    np.save(stream, np.ones((2,2),np.uint32))
    payload = stream.getvalue()
    compression = zipfile.ZIP_DEFLATED if kind == 'crc_deflated' else zipfile.ZIP_STORED
    with zipfile.ZipFile(path,'w',compression=compression) as z:
        z.writestr('labels.npy', payload)
    raw = bytearray(path.read_bytes())
    local = raw.index(b'PK\x03\x04')
    central = raw.index(b'PK\x01\x02')
    if kind.startswith('crc'):
        raw[central+16] ^= 1
    elif kind == 'truncated':
        raw = raw[:central]
    elif kind == 'encrypted':
        raw[local+6] |= 1
        raw[central+8] |= 1
    elif kind == 'unsupported_compression':
        raw[local+8:local+10] = (99).to_bytes(2,'little')
        raw[central+10:central+12] = (99).to_bytes(2,'little')
    elif kind == 'member_header':
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('labels.npy', b'\x93NUMPY')
        return
    path.write_bytes(raw)


@pytest.mark.parametrize('kind',['crc_stored','crc_deflated','truncated','encrypted','unsupported_compression','member_header'])
def test_bad_raw_npz_api_cli(tmp_path,kind):
    root = tmp_path/'bundle'
    metadata = valid(root)
    malformed_archive(root/'raw.npz',kind)
    commit(root,metadata)  # Outer SHA is consistent; exercise the format parser.
    with pytest.raises(StitchTraceError, match='INVALID_CACHE'):
        Bundle.open(root)
    cli_reject(root,'INVALID_CACHE')


def test_valid_compressed_npz(tmp_path):
    root = tmp_path/'bundle'
    metadata = valid(root)
    np.savez_compressed(root/'raw.npz',labels=np.ones((2,2),np.uint32))
    commit(root,metadata)
    Bundle.open(root)
