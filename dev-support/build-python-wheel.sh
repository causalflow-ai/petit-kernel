#!/bin/sh

set -eu

ROCM_ARCH="${ROCM_ARCH:-gfx90a;gfx942;gfx950}"

if [ -z "${TORCH_PATH:-}" ]; then
    TORCH_PATH=$(
        python3 - <<'PY'
import pathlib
import torch

print(pathlib.Path(torch.__file__).resolve().parent)
PY
    )
fi

if [ -f "${TORCH_PATH}/share/cmake/Torch/TorchConfig.cmake" ]; then
    TORCH_CMAKE_PREFIX="${TORCH_PATH}"
elif [ -f "${TORCH_PATH}/torch/share/cmake/Torch/TorchConfig.cmake" ]; then
    TORCH_CMAKE_PREFIX="${TORCH_PATH}/torch"
elif [ -f "${TORCH_PATH}/TorchConfig.cmake" ]; then
    TORCH_CMAKE_PREFIX="${TORCH_PATH}"
else
    echo "Unable to find TorchConfig.cmake under TORCH_PATH=${TORCH_PATH}" >&2
    echo "Set TORCH_PATH to the torch package directory or site-packages directory." >&2
    exit 1
fi

CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/rocm;${TORCH_CMAKE_PREFIX}}"
CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH} -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} -DGPU_TARGETS=${ROCM_ARCH}"

if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf is required for mandatory auditwheel repair." >&2
    exit 1
fi

export CMAKE_ARGS
python3 setup.py clean --all
python3 setup.py bdist_wheel --dist-dir=dist
python3 -m auditwheel repair dist/petit_kernel-*-cp*-cp*-linux_x86_64.whl --exclude '*' -w dist/
