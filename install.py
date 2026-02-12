import subprocess
import sys
import os


def get_cupy_package():
    """Detect PyTorch's CUDA version and return the matching cupy package name."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[BIM-VFI] WARNING: CUDA not available. cupy requires CUDA.")
            return None
        cuda_version = torch.version.cuda
        if cuda_version is None:
            print("[BIM-VFI] WARNING: PyTorch has no CUDA version info.")
            return None
        major = int(cuda_version.split(".")[0])
        cupy_pkg = f"cupy-cuda{major}x"
        print(f"[BIM-VFI] Detected CUDA {cuda_version}, will use {cupy_pkg}")
        return cupy_pkg
    except Exception as e:
        print(f"[BIM-VFI] WARNING: Could not detect CUDA version: {e}")
        return None


def update_requirements(cupy_pkg):
    """Write the correct cupy package into requirements.txt."""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    lines = []
    if os.path.exists(requirements_path):
        with open(requirements_path, "r") as f:
            lines = [l.rstrip() for l in f if not l.strip().startswith("cupy")]
    if cupy_pkg and cupy_pkg not in lines:
        lines.append(cupy_pkg)
    with open(requirements_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def install():
    cupy_pkg = get_cupy_package()
    if cupy_pkg:
        update_requirements(cupy_pkg)

    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", requirements_path
    ])


if __name__ == "__main__":
    install()
