from pathlib import Path
import json
import pytest
from stitchtrace import Bundle,StitchTraceError
from stitchtrace.cli import main
from stitchtrace.storage import safe_path,atomic_bytes


@pytest.mark.parametrize('name',['../secret','/absolute','a/../b','a//b','a\\b',''])
def test_unsafe_member(tmp_path,name):
    with pytest.raises(StitchTraceError):safe_path(tmp_path,name)


def test_duplicate_json_and_cli_exit(tmp_path,capsys):
    (tmp_path/'manifest.json').write_text('{"schema":1,"schema":1,"files":{}}')
    with pytest.raises(StitchTraceError,match='DUPLICATE_KEY'):Bundle.open(tmp_path)
    assert main(['verify',str(tmp_path)])==2
    assert 'DUPLICATE_KEY' in capsys.readouterr().err


def test_missing_manifest_coded(tmp_path):
    with pytest.raises(StitchTraceError,match='INVALID_BUNDLE'):Bundle.open(tmp_path)
