"""Explicit cache preparation. Networking requires --download."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile

P = Path(__file__).resolve().parents[1]

def prepare(cache_dir, download=False, core=False):
    cache = Path(cache_dir).expanduser().resolve()
    env = os.environ.copy()
    env['UV_CACHE_DIR'] = str(cache)
    env['UV_PYTHON_DOWNLOADS'] = 'never'
    lock = P / ('requirements-core.lock' if core else 'requirements-capture.lock')
    with tempfile.TemporaryDirectory(prefix='stitchtrace-preflight-') as temp:
        target = Path(temp) / 'venv'
        commands = [
            ['uv', 'venv', '--python', sys.executable, str(target)],
            ['uv', 'pip', 'install', '--python', str(target/'bin/python'),
             '--index-url', 'https://pypi.org/simple', '-r', str(lock)]
        ]
        if not download:
            commands[1].append('--offline')
        for cmd in commands:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr.strip())
    return {'status': 'ready', 'cache_dir': str(cache), 'lock': lock.name,
            'mode': 'explicit-download' if download else 'offline-install-verified'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--download', action='store_true', help='Explicitly allow official PyPI dependency downloads')
    parser.add_argument('--core', action='store_true', help='Prepare only the minimal core lock')
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.cache_dir, args.download, args.core)))
    except (RuntimeError, OSError) as error:
        print(json.dumps({'error':'DEPENDENCY_PREFLIGHT', 'detail':str(error)}), file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
