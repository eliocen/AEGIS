# AEGIS Research Core v1.0 鈥?Environment and Dependency Ledger

| Element | Frozen/documented value | Classification |
|---|---|---|
| OS family | Windows | Primary research environment context |
| Shell | PowerShell | Primary research environment context |
| Repository root | `D:\Research\AEGIS` | Primary workstation path; not a portable requirement |
| Python | 3.10.11 | Frozen primary runtime context |
| Python executable | `D:\Research\AEGIS\.venv\Scripts\python.exe` | Primary workstation path |
| GPU | NVIDIA GeForce GTX 1660 SUPER | Frozen primary hardware context |
| PyTorch | 2.13.0+cu132 | Frozen primary runtime context |
| CUDA available | True | Frozen primary runtime observation |
| Fakeddit train cache | `data/processed/fakeddit/frozen_embeddings/train_n5000_seed42` | Frozen v0.35 data boundary |
| Fakeddit validation cache | `data/processed/fakeddit/frozen_embeddings/validation_n1000_seed42` | Frozen v0.35 data boundary |

## Dependency interpretation

The exact research environment and a portable minimum environment are different concepts. This ledger records the former where frozen evidence exists. Repository dependency/configuration files should be used to reconstruct package requirements; P3 does not invent versions that are absent from frozen evidence.

## Determinism boundary

Exact numerical reproduction may be affected by GPU architecture, CUDA/cuDNN versions and algorithms, PyTorch builds, nondeterministic kernels, driver versions, filesystem behavior, and third-party model/data assets. Any independent rerun should record these values rather than silently treating them as equivalent.

## Data integrity

Train manifest SHA256: `5BC8BF53F988B3DD5299A1D940DB6676F48E37250F52C8B3569150BE153417C5`.

Validation manifest SHA256: `C4E95F818C24043DB1DEE9C0CFC0DE85C2E591E9E77B98CB2490046D1BC04F87`.

Official-test access: **0 samples**.
