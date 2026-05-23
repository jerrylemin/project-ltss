# CUDA Setup

## Current Status

- GPU detected by `nvidia-smi`: NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB VRAM.
- Driver version: 591.86.
- CUDA Toolkit installed with `winget install --id Nvidia.CUDA -e`.
- Toolkit path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`.
- `nvcc -V`: CUDA compilation tools release 13.2, V13.2.78.
- Python CUDA package: `numba-cuda==0.30.2`.
- `from numba import cuda; cuda.is_available()`: `True` after loading the CUDA Toolkit path.

## Commands Used

```powershell
winget search Nvidia.CUDA
winget install --id Nvidia.CUDA -e --accept-source-agreements --accept-package-agreements
.\.venv\Scripts\python.exe -m pip install numba-cuda
powershell -ExecutionPolicy Bypass -File scripts/check_cuda.ps1
```

## Terminal Reload Note

New terminals should pick up `CUDA_PATH` from machine environment. Existing terminals may need:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:Path = "$env:CUDA_PATH\bin;$env:CUDA_PATH\bin\x64;$env:Path"
```

The project scripts do this automatically by searching `CUDA_PATH` and the default NVIDIA Toolkit install directory.

## Troubleshooting

- If `nvcc` is not found, run `powershell -ExecutionPolicy Bypass -File scripts/check_cuda.ps1` and check `artifacts/environment_check.json`.
- If `numba-cuda` import fails with `cudart*.dll` missing, install CUDA Toolkit or make sure the toolkit `bin` directory is in PATH.
- If future kernels require compiling external CUDA code with `nvcc`, Visual Studio Build Tools may be required. The current Numba kernels do not need a C++ build step.
