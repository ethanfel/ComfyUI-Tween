import os
import subprocess
import sys


def install():
    """Install required dependencies without mutating optional GPU packages."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", requirements_path
    ])
    print(
        "[Tween] Optional cupy is not installed automatically. "
        "BIM-VFI, SGM-VFI, and GIMM-VFI use the PyTorch fallback unless you "
        "install the matching cupy wheel manually; EMA-VFI, SPEED, and "
        "LDF-VFI do not use cupy."
    )


if __name__ == "__main__":
    install()
