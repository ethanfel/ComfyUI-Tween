import subprocess
import sys
import os


def get_cupy_package():
    """Detect PyTorch's CUDA version and return the matching cupy package name."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        cuda_version = torch.version.cuda
        if cuda_version is None:
            return None
        major = int(cuda_version.split(".")[0])
        cupy_pkg = f"cupy-cuda{major}x"
        return cupy_pkg
    except Exception:
        return None


def install():
    # Install core requirements first
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", requirements_path
    ])

    # Try to install cupy for NVIDIA users (optional, improves performance)
    cupy_pkg = get_cupy_package()
    if cupy_pkg:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", cupy_pkg
            ])
            print(f"[Tween] cupy installed ({cupy_pkg}) — fast CUDA kernels enabled")
        except subprocess.CalledProcessError:
            print(f"[Tween] WARNING: Could not install {cupy_pkg}. "
                  f"BIM-VFI, SGM-VFI, and GIMM-VFI will use slower PyTorch fallback.")
    else:
        print("[Tween] cupy skipped (no NVIDIA CUDA). "
              "BIM-VFI, SGM-VFI, and GIMM-VFI will use PyTorch fallback.")


if __name__ == "__main__":
    install()
