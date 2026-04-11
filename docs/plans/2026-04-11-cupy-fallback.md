# Pure-PyTorch Fallbacks for cupy Kernels

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BIM-VFI, SGM-VFI, and GIMM-VFI work without cupy by adding pure-PyTorch fallback implementations of softsplat and costvol.

**Architecture:** Each kernel file (`sgm_vfi_arch/softsplat.py`, `gimm_vfi_arch/.../softsplat.py`, `bim_vfi_arch/costvol.py`) gets a `_pytorch_*` fallback function. The `softsplat_func.forward()` and `costvol_func.forward()` methods dispatch to cupy when available, otherwise use the fallback. The `_check_cupy()` gate in `nodes.py` is removed so models can load on any backend.

**Tech Stack:** PyTorch (`scatter_add_`, `F.unfold`, `F.pad`)

---

### Task 1: Add pure-PyTorch softsplat fallback to SGM-VFI

**Files:**
- Modify: `sgm_vfi_arch/softsplat.py`

**Step 1: Add cupy availability flag and fallback function**

At the top of `sgm_vfi_arch/softsplat.py`, change the hard `import cupy` to a try/except, and add the fallback function after the `cuda_launch` function (before the `softsplat()` function).

Replace:
```python
import cupy
```
With:
```python
try:
    import cupy
except ImportError:
    cupy = None
```

Add this fallback function (after `cuda_launch`, before `softsplat`):
```python
def _pytorch_softsplat(tenIn, tenFlow):
    B, C, H, W = tenIn.shape
    tenOut = tenIn.new_zeros(B, C, H, W)

    # Build base grid: (x, y) for each pixel
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=tenIn.device, dtype=tenIn.dtype),
        torch.arange(W, device=tenIn.device, dtype=tenIn.dtype),
        indexing='ij',
    )

    # Target positions
    flt_x = grid_x.unsqueeze(0) + tenFlow[:, 0, :, :]  # (B, H, W)
    flt_y = grid_y.unsqueeze(0) + tenFlow[:, 1, :, :]

    # Filter non-finite
    valid = torch.isfinite(flt_x) & torch.isfinite(flt_y)
    flt_x = torch.where(valid, flt_x, torch.zeros_like(flt_x))
    flt_y = torch.where(valid, flt_y, torch.zeros_like(flt_y))

    # Four neighbors (NW, NE, SW, SE)
    nw_x = flt_x.floor().long()
    nw_y = flt_y.floor().long()

    # Bilinear weights
    frac_x = flt_x - nw_x.float()
    frac_y = flt_y - nw_y.float()
    w_nw = (1.0 - frac_x) * (1.0 - frac_y)
    w_ne = frac_x * (1.0 - frac_y)
    w_sw = (1.0 - frac_x) * frac_y
    w_se = frac_x * frac_y

    # Zero out invalid pixels
    w_nw = w_nw * valid
    w_ne = w_ne * valid
    w_sw = w_sw * valid
    w_se = w_se * valid

    # For each of the 4 neighbors, scatter into output
    for dx, dy, w in [(0, 0, w_nw), (1, 0, w_ne), (0, 1, w_sw), (1, 1, w_se)]:
        tx = nw_x + dx
        ty = nw_y + dy
        in_bounds = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
        w_masked = w * in_bounds

        # Flatten to 1D index for scatter_add
        idx = (ty.clamp(0, H - 1) * W + tx.clamp(0, W - 1))  # (B, H, W)
        idx = idx.unsqueeze(1).expand_as(tenIn)  # (B, C, H, W)
        weighted = tenIn * w_masked.unsqueeze(1)  # (B, C, H, W)
        tenOut.view(B, C, -1).scatter_add_(2, idx.reshape(B, C, -1), weighted.reshape(B, C, -1))

    return tenOut
```

**Step 2: Update softsplat_func.forward to use fallback**

In `softsplat_func.forward()`, replace the `elif tenIn.is_cuda != True: assert(False)` block so it dispatches to the fallback when cupy is unavailable or when not on CUDA:

```python
# Current:
        if tenIn.is_cuda == True:
            cuda_launch(cuda_kernel(...))(...) 
        elif tenIn.is_cuda != True:
            assert(False)

# New:
        if tenIn.is_cuda and cupy is not None:
            cuda_launch(cuda_kernel(...))(...) 
        else:
            tenOut = _pytorch_softsplat(tenIn, tenFlow)
```

Also guard the `@cupy.memoize` decorator on `cuda_launch`:
```python
# Current:
@cupy.memoize(for_each_device=True)
def cuda_launch(strKey:str):

# New:
def cuda_launch(strKey:str):
```
(The function already has its own dict-based caching via `objCudacache`, and the memoize is redundant anyway. But the real issue is it crashes at import when cupy=None.)

Wait - actually `cuda_launch` uses `cupy.RawKernel` inside, so it's only ever called on the cupy path. The `@cupy.memoize` decorator is the problem: it runs at import time. Replace it:

```python
# Replace @cupy.memoize(for_each_device=True) with a simple cache dict
_cuda_launch_cache = {}

def cuda_launch(strKey:str):
    if strKey not in _cuda_launch_cache:
        if 'CUDA_HOME' not in os.environ:
            os.environ['CUDA_HOME'] = cupy.cuda.get_cuda_path()
        _cuda_launch_cache[strKey] = cupy.RawKernel(
            objCudacache[strKey]['strKernel'],
            objCudacache[strKey]['strFunction'],
            options=tuple(['-I ' + os.environ['CUDA_HOME'],
                           '-I ' + os.environ['CUDA_HOME'] + '/include'])
        )
    return _cuda_launch_cache[strKey]
```

**Step 3: Commit**

```bash
git add sgm_vfi_arch/softsplat.py
git commit -m "feat: add pure-PyTorch softsplat fallback for SGM-VFI"
```

---

### Task 2: Add pure-PyTorch softsplat fallback to GIMM-VFI

**Files:**
- Modify: `gimm_vfi_arch/generalizable_INR/modules/softsplat.py`

**Step 1: Add cupy availability flag and fallback function**

Same pattern as Task 1. Replace `import cupy` with try/except. Add the same `_pytorch_softsplat()` function. Replace `@cupy.memoize(for_each_device=True)` on `cuda_launch` with a dict cache.

The GIMM softsplat.py already has `@torch.compiler.disable()` on `cuda_launch` — keep that decorator.

**Step 2: Update softsplat_func.forward dispatch**

Same pattern: if `tenIn.is_cuda and cupy is not None` → cupy path, else → `_pytorch_softsplat`.

**Step 3: Commit**

```bash
git add gimm_vfi_arch/generalizable_INR/modules/softsplat.py
git commit -m "feat: add pure-PyTorch softsplat fallback for GIMM-VFI"
```

---

### Task 3: Add pure-PyTorch costvol fallback to BIM-VFI

**Files:**
- Modify: `bim_vfi_arch/costvol.py`

**Step 1: Add the fallback function**

After the existing `cuda_launch` function, add:

```python
def _pytorch_costvol(tenOne, tenTwo, intKernelSize):
    B, C, H, W = tenOne.shape
    pad = (intKernelSize - 1) // 2

    # Pad tenTwo with zeros so out-of-bounds accesses yield 0 (matches CUDA kernel)
    tenTwo_padded = F.pad(tenTwo, [pad, pad, pad, pad])

    # Unfold into (B, C, K*K, H, W) patches
    patches = tenTwo_padded.unfold(2, intKernelSize, 1).unfold(3, intKernelSize, 1)
    # patches shape: (B, C, H, W, K, K)
    patches = patches.contiguous().view(B, C, H, W, intKernelSize * intKernelSize)
    # -> (B, C, H, W, K^2)

    # Dot product: sum over C
    # tenOne: (B, C, H, W) -> (B, C, H, W, 1)
    tenOut = (tenOne.unsqueeze(-1) * patches).sum(dim=1)
    # tenOut: (B, H, W, K^2)

    # Permute to (B, K^2, H, W) to match CUDA output layout
    tenOut = tenOut.permute(0, 3, 1, 2).contiguous()

    return tenOut
```

Add `import torch.nn.functional as F` at the top if not already present.

**Step 2: Update costvol_func.forward dispatch**

The current forward unconditionally calls `cuda_launch(cuda_kernel(...))`. Change to:

```python
@staticmethod
@torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
def forward(self, tenOne, tenTwo, intKernelSize):
    if tenOne.is_cuda and cupy is not None:
        # existing cupy code (unchanged)
        tenOut = tenOne.new_empty([tenOne.shape[0], intKernelSize ** 2, tenOne.shape[2], tenOne.shape[3]])
        cuda_launch(cuda_kernel(...))(...) 
    else:
        tenOut = _pytorch_costvol(tenOne, tenTwo, intKernelSize)

    self.save_for_backward(tenOne, tenTwo)
    self.intKernelSize = intKernelSize
    return tenOut
```

**Step 3: Commit**

```bash
git add bim_vfi_arch/costvol.py
git commit -m "feat: add pure-PyTorch costvol fallback for BIM-VFI"
```

---

### Task 4: Remove _check_cupy gate from nodes.py

**Files:**
- Modify: `nodes.py`

**Step 1: Remove the _check_cupy function and all its call sites**

Delete the `_check_cupy()` function definition (lines 22-41). Remove the three calls:
- Line 209: `_check_cupy("BIM-VFI")` (in BIM-VFI load)
- Line 1377: `_check_cupy("SGM-VFI")` (in SGM-VFI load)
- Line 1804: `_check_cupy("GIMM-VFI")` (in GIMM-VFI load)

**Step 2: Commit**

```bash
git add nodes.py
git commit -m "feat: remove cupy requirement gate, models now fallback to pure PyTorch"
```

---

### Task 5: Make install.py not force cupy installation

**Files:**
- Modify: `install.py`

**Step 1: Change cupy from required to optional**

Make cupy a soft dependency — try to install it but don't fail if it can't be installed (ROCm users, no CUDA toolkit, etc.). Change `install()`:

```python
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
            print(f"[Tween] cupy installed successfully ({cupy_pkg})")
        except subprocess.CalledProcessError:
            print(f"[Tween] WARNING: Could not install {cupy_pkg}. "
                  f"BIM-VFI, SGM-VFI, and GIMM-VFI will use slower PyTorch fallback.")
    else:
        print("[Tween] cupy not available (no NVIDIA CUDA). "
              "BIM-VFI, SGM-VFI, and GIMM-VFI will use PyTorch fallback.")
```

Also stop writing cupy into `requirements.txt` — remove the `update_requirements` call and function.

**Step 2: Commit**

```bash
git add install.py
git commit -m "feat: make cupy optional in install.py"
```
