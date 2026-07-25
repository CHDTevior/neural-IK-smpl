#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMPLIK_ENV_PREFIX="${SMPLIK_ENV_PREFIX:-/iridisfs/scratch/ts1v23/workspace/learned_IK/.envs/smplik-py38}"

mkdir -p "$(dirname "$SMPLIK_ENV_PREFIX")"

if [[ ! -x "$SMPLIK_ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$SMPLIK_ENV_PREFIX" \
    --override-channels \
    -c pytorch3d -c pytorch -c defaults -c conda-forge \
    python=3.8.20 \
    pip=21.3.1 \
    setuptools=59.5.0 \
    wheel=0.37.1 \
    numpy=1.21.6 \
    'pytorch=1.9.0=py3.8_cuda11.1_cudnn8.0.5_0' \
    'torchvision=0.10.0=py38_cu111' \
    cudatoolkit=11.1.1 \
    'pytorch3d=0.6.2=py38_cu111_pyt190' \
    fvcore=0.1.5.post20221221 \
    iopath=0.1.10
fi

"$SMPLIK_ENV_PREFIX/bin/python" -m pip install \
  --constraint "$REPO_DIR/ops/constraints.compat.txt" \
  pytorch-lightning==1.4.7 \
  pandas==1.1.4 \
  pyarrow==4.0.0 \
  scikit-learn==0.23.2 \
  hydra-core==1.0.4 \
  GitPython==3.1.11 \
  plotly==5.6.0 \
  onnx==1.11.0 \
  chumpy==0.70 \
  'smplx[all]==0.1.28' \
  vector-quantize-pytorch==0.2.2 \
  h5py==3.7.0 \
  torchmetrics==0.4.0 \
  roma==1.2.5 \
  markupsafe==2.0.1 \
  tensorboard==2.11.2

"$SMPLIK_ENV_PREFIX/bin/python" -m pip check
