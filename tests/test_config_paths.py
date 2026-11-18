"""M7: placeholder data roots give a clear message; importing piepy never depends on the cwd."""

import subprocess
import sys

import pytest

from piepy.core.errors import SessionNotFoundError
from piepy.core.paths.locator import SessionLocator


def test_missing_data_root_is_named_as_placeholder(tmp_path):
    loc = SessionLocator(behaviour_roots=[("presentation", str(tmp_path / "data/presentation"))])
    with pytest.raises(SessionNotFoundError, match="placeholders"):
        loc._resolve("240810_KC150_detect__no_cam_KC")


def test_existing_root_keeps_plain_message(tmp_path):
    loc = SessionLocator(behaviour_roots=[("presentation", str(tmp_path))])
    with pytest.raises(SessionNotFoundError) as e:
        loc._resolve("240810_KC150_detect__no_cam_KC")
    assert "placeholders" not in str(e.value)


def test_import_works_from_filesystem_root():
    r = subprocess.run([sys.executable, "-c", "import piepy.core.config"], cwd="/", capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
