# flash-mHC

`flash-mHC` is a standalone, production-focused extraction of the mHC-lite Torch+Triton path.

It provides:
- Triton fused kernels for mHC-lite K1/K3/K4 steps
- `torch.library` custom ops compatible with `torch.compile`
- A reusable `MHCLiteBlock` wrapper module
- Kernel microbenchmark + one-shot launch-parameter grid search
- CUDA correctness tests against PyTorch references

## Status

- Current fused kernels are specialized for `n_streams=4`.
- Triton path is used for CUDA + bf16 inputs with shape `(T, n, C)`.
- Fallback path remains pure PyTorch in `MHCLiteBlock`.
- Runtime Triton autotune is enabled by default for K1/K3/K4 kernels (capped config sets + Triton disk cache).
- `MHCLiteBlock` supports optional activation checkpointing and disables autotune during eval/inference by default.

## Install

```bash
pip install -e .
# optional test tooling
pip install -e .[dev]
```

## Quick Start

```python
import torch
import torch.nn as nn
from flash_mhc import MHCLiteBlock

layer = nn.Linear(1024, 1024, bias=False).cuda().bfloat16()
block = MHCLiteBlock(
    n_streams=4,
    hidden_size=1024,
    layer=layer,
    triton_fused=True,
    activation_checkpointing=True,
).cuda().bfloat16()

x = torch.randn(65536, 4, 1024, device="cuda", dtype=torch.bfloat16)
out = block(x)
print(out.shape)  # (65536, 4, 1024)
```

## Benchmarks

Kernel microbenchmark:

```bash
python scripts/benchmark_kernels.py --T 65536 --C 1024 --n 4 --peak-gbps 1792
```

Grid search for launch parameters:

```bash
python scripts/gridsearch.py --T 65536 --C 1024 --n 4 --json-out output/gridsearch_sm120.json
```

## Autotune Controls

By default, `flash-mHC` uses Triton's runtime autotune wrappers for fused kernels.

- `FLASH_MHC_TRITON_AUTOTUNE=0`: disable runtime autotune and use legacy hardcoded launch params from `ops.py`.
- `FLASH_MHC_TRITON_AUTOTUNE_STATUS=0`: silence one-time autotune status logs.
- `TRITON_CACHE_DIR=/path/to/cache`: override Triton disk cache location (default is Triton's normal cache path).

## Tests

```bash
pytest -q tests/test_correctness.py
```

## Layout

- `src/flash_mhc/kernels.py`: Triton kernels
- `src/flash_mhc/ops.py`: `torch.library` op registration + autograd
- `src/flash_mhc/block.py`: standalone `MHCLiteBlock`
- `scripts/benchmark_kernels.py`: per-kernel timing and efficiency
- `scripts/gridsearch.py`: one-shot launch-parameter sweep
- `tests/test_correctness.py`: CUDA correctness tests
- `docs/ARCHITECTURE.md`: kernel/dataflow details
- `docs/TUNING_SM120.md`: SM120 tuning summary and selected params
- `docs/PUBLISHING.md`: packaging and release checklist
