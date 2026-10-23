"""StitchTrace: record, replay and attribute tile-assembly decisions of instance segmentation."""
from .bundle import Bundle, canonical, measurements
from .replay import replay
from .policies import register_policy, available_policies, run_policy
from .compare import compare, attribute
from .errors import StitchTraceError
__version__ = '1.0.0'
__all__ = ['Bundle', 'canonical', 'measurements', 'replay', 'register_policy', 'available_policies',
           'run_policy', 'compare', 'attribute', 'StitchTraceError']
