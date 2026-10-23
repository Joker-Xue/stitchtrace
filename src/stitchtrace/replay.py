import numpy as np
from .bundle import Bundle, canonical, measurements
from .errors import require
from .policies import run_policy


def replay(bundle, policy='native', **params):
    """Replay recorded decisions under ``policy`` without inference; return labels and measurements.

    ``bundle`` is a path or an already verified ``Bundle``. Native replay must reproduce the
    recorded output partition exactly.
    """
    b = bundle if isinstance(bundle, Bundle) else Bundle.open(bundle)
    output = run_policy(b, policy, **params)
    if policy == 'native':
        require(np.array_equal(output, canonical(b.array('expected.npy'))), 'REPLAY_MISMATCH', 'partition differs from original')
    return output, measurements(output, b.meta['spacing'])
