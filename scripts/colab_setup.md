# Running on Google Colab (T4, sm_75)

The T4 is Turing like the RTX 2060, so the same `CMAKE_CUDA_ARCHITECTURES 75`
binary runs on both. Note that Colab gives you only ~2 vCPUs, so **CPU baseline
numbers from Colab are not comparable** to a desktop CPU; use Colab for
correctness and GPU numbers, and label them with the device in the CSV.

## 1. Select a GPU runtime

*Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.*

## 2. Check the toolchain

```bash
!nvidia-smi
!nvcc --version
!cmake --version
```

If `cmake --version` is below 3.24:

```bash
!pip install -q --upgrade cmake
```

## 3. Clone and build

`!cd` runs in a throwaway subshell, so use the `%cd` magic to change directory.

```bash
!git clone https://github.com/<you>/cuda-mc-pricer.git
%cd cuda-mc-pricer
!cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
!cmake --build build -j
```

## 4. Test

```bash
!ctest --test-dir build --output-on-failure
```

GPU tests print `[SKIP] ... not implemented yet` until you flip `kImplemented`
in the `.cu` files.

## 5. Benchmark

```bash
!./build/mc_pricer --impl all --paths 1e6
!./build/mc_pricer --impl gpu_opt --paths 1e6,1e7,1e8,1e9 --runs 5 --csv results/colab_t4.csv
```

## 6. Plot and download

```bash
!python scripts/plot_results.py results/colab_t4.csv --out results/colab_t4.png
```

```python
from google.colab import files
files.download("results/colab_t4.csv")
files.download("results/colab_t4.png")
```

## 7. Profiling (optional)

```bash
!ncu --kernel-name price_paths_opt --launch-count 1 --section SpeedOfLight --section Occupancy \
     ./build/mc_pricer --impl gpu_opt --paths 1e8 --runs 1
```

If `ncu` reports `ERR_NVGPUCTRPERM`, the VM doesn't expose performance counters
to you; profile on your local RTX 2060 instead (see README → Profiling).

## Iterating

After editing a `.cu` file locally and pushing:

```bash
!git pull && cmake --build build -j && ./build/mc_pricer --impl gpu_opt --paths 1e7
```

Or edit directly in Colab's file browser (left sidebar), then rebuild.
