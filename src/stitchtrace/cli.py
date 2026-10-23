import argparse
import json
import sys
from pathlib import Path
import numpy as np
from . import Bundle, replay, measurements, compare, attribute, available_policies, StitchTraceError
from .policies import block_seams
from .storage import atomic_array


def _params(pairs):
    out = {}
    for item in pairs or []:
        key, sep, value = item.partition('=')
        if not sep:
            raise StitchTraceError('INVALID_PARAMETER', item)
        out[key] = float(value)
    return out


def _labels(path):
    return np.load(path, allow_pickle=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='stitchtrace')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'inspect', 'replay'):
        p = sub.add_parser(name)
        p.add_argument('bundle')
        if name == 'replay':
            p.add_argument('--policy', default='native')
            p.add_argument('--param', action='append', metavar='KEY=VALUE', help='numeric policy parameter')
            p.add_argument('--output', required=True)
    sub.add_parser('policies')
    p = sub.add_parser('measure')
    p.add_argument('array')
    p.add_argument('--spacing', nargs='+', type=float)
    p = sub.add_parser('compare', help='object-level comparison of two label arrays (.npy)')
    p.add_argument('reference')
    p.add_argument('candidate')
    p.add_argument('--iou', type=float, default=0.5)
    p.add_argument('--bundle', help='attribute differences to the block seams of this bundle')
    args = parser.parse_args(argv)
    try:
        if args.command == 'measure':
            result = measurements(_labels(args.array), args.spacing)
        elif args.command == 'policies':
            result = available_policies()
        elif args.command == 'compare':
            ref, cand = _labels(args.reference), _labels(args.candidate)
            if args.bundle:
                result = attribute(ref, cand, block_seams(Bundle.open(args.bundle)), args.iou)
            else:
                result = compare(ref, cand, args.iou)
        else:
            b = Bundle.open(args.bundle)
            if args.command == 'replay':
                if Path(args.output).exists():
                    raise StitchTraceError('OUTPUT_EXISTS', args.output)
                a, result = replay(b, args.policy, **_params(args.param))
                atomic_array(args.output, a)
            else:
                result = b.meta if args.command == 'inspect' else {'verified': True, 'schema': 1, 'backend': b.meta['backend']}
        print(json.dumps(result, indent=2))
        return 0
    except (StitchTraceError, OSError, ValueError, KeyError, TypeError) as e:
        print(json.dumps({'error': getattr(e, 'code', 'INVALID_INPUT'), 'message': str(e)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
