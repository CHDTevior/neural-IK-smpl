#!/usr/bin/env bash
# Idempotent setup: clone upstream SMPL-IK at the pinned commit and overlay ops/.
#
# This public repo is an OVERLAY. It contains none of the upstream source, no
# SMPL body models, no datasets, and no checkpoints. This script materializes
# the working tree:
#
#   ./smpl-ik/         upstream boreshkinai/smpl-ik @ ea74d47 (non-commercial
#                      academic license -- see its LICENSE.md)
#   ./smpl-ik/ops/     our infra, copied from ./ops/
#
# Fail-loud: any error aborts the script.

set -euo pipefail

UPSTREAM_URL="https://github.com/boreshkinai/smpl-ik.git"
UPSTREAM_COMMIT="ea74d47f8adafed7af95f9c8eac4de6a715bf0bc"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="$HERE/smpl-ik"

if [[ ! -d "$HERE/ops" ]]; then
  echo "ERROR: $HERE/ops not found; run this script from the repo root" >&2
  exit 1
fi

# 1) Clone (or verify) the upstream checkout at the pinned commit.
if [[ -d "$CLONE_DIR/.git" ]]; then
  echo "Upstream clone already exists: $CLONE_DIR"
else
  git clone "$UPSTREAM_URL" "$CLONE_DIR"
fi
git -C "$CLONE_DIR" fetch --quiet origin "$UPSTREAM_COMMIT" 2>/dev/null || true
git -C "$CLONE_DIR" checkout --quiet "$UPSTREAM_COMMIT"
ACTUAL="$(git -C "$CLONE_DIR" rev-parse HEAD)"
if [[ "$ACTUAL" != "$UPSTREAM_COMMIT" ]]; then
  echo "ERROR: upstream checkout is at $ACTUAL, expected $UPSTREAM_COMMIT" >&2
  exit 1
fi
echo "Upstream pinned at $UPSTREAM_COMMIT"

# 2) Overlay our ops/ into the checkout (rsync if available, cp otherwise).
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete "$HERE/ops/" "$CLONE_DIR/ops/"
else
  rm -rf "$CLONE_DIR/ops"
  cp -a "$HERE/ops" "$CLONE_DIR/ops"
fi
echo "Overlaid $(find "$CLONE_DIR/ops" -type f | wc -l) files into $CLONE_DIR/ops/"

cat <<'EOF'

Next steps (details in README.md):
  1. Create the pinned conda environment           -> README "Setup"
  2. Register + download SMPL body models & AMASS  -> README "Data preparation"
  3. Preprocess AMASS + build the Feather caches   -> README "Data preparation"
     (WARNING: cache build peaked at 389.79 GiB RSS; use a >=800 GiB-RAM node)
  4. Train:
       canonical:  cd smpl-ik && python run.py --config=configs/experiments/smplik_amass.yaml
       multi-GPU:  bash smpl-ik/ops/launch_hopper_variant.sh {h100x7|h200x4} {smoke|formal}
                   (adapt the cluster-specific paths/job IDs inside first; see docs/MULTI_GPU.md)
  5. Or skip training: download the pretrained checkpoint (README "Pretrained weights")
     and run the interactive demo: SMPLIK_CKPT=... python demo/server.py
EOF
