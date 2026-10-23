import importlib.util
from unittest.mock import patch
import pytest
from stitchtrace import StitchTraceError


@pytest.mark.skipif(importlib.util.find_spec('stardist') is None,reason='optional installed-backend test')
def test_version_source_signature_refusal():
    from stitchtrace.backends.common import compatible
    compatible('stardist')
    with patch('stitchtrace.backends.common.importlib.metadata.version',return_value='0.unsupported'):
        with pytest.raises(StitchTraceError,match='VERSION_UNSUPPORTED'):compatible('stardist')
    with patch('stitchtrace.backends.common.digest',return_value='mismatch'):
        with pytest.raises(StitchTraceError,match='SOURCE_UNSUPPORTED'):compatible('stardist')
    with patch('stitchtrace.backends.common.inspect.signature',return_value='(changed)'):
        with pytest.raises(StitchTraceError,match='SIGNATURE_UNSUPPORTED'):compatible('stardist')
