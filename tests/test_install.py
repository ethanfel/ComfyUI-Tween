import os
import subprocess
import sys

import install as tween_install


def test_installer_never_installs_optional_cupy(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "check_call", calls.append)

    tween_install.install()

    requirements_path = os.path.join(
        os.path.dirname(tween_install.__file__), "requirements.txt"
    )
    assert calls == [[
        sys.executable, "-m", "pip", "install", "-r", requirements_path
    ]]
