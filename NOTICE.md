# Third-party notices

This repository (the overlay: `ops/`, `demo/server.py`, `demo/run_demo_inference.py`,
`demo/frontend.html`, `docs/`, `setup.sh`) is MIT-licensed (see `LICENSE`).
Everything below is third-party material with its own license terms. None of it
is redistributed here except the three vendored three.js modules noted below.

## SMPL-IK (upstream code — NOT included)

- Repository: <https://github.com/boreshkinai/smpl-ik> (pinned commit `ea74d47f8adafed7af95f9c8eac4de6a715bf0bc`)
- Paper: Voleti et al., *SMPL-IK: Learned Morphology-Aware Inverse Kinematics for AI Driven Artistic Workflows*, arXiv:2208.08274
- License: upstream `LICENSE.md` permits use **only for non-commercial academic
  research / reference purposes**. We do not redistribute any upstream file;
  `setup.sh` clones it directly from the upstream repository and `.gitignore`
  keeps the clone out of this repo. Anything you build on top of the combined
  tree inherits that non-commercial restriction.

## ProtoRes lineage

SMPL-IK extends ProtoRes (Oreshkin et al., *ProtoRes: Proto-Residual Network
for Pose Authoring via Learned Inverse Kinematics*, ICLR 2022,
arXiv:2106.01981). The backbone/architecture concepts used by the upstream
code originate there; the same Unity Technologies non-commercial research
posture applies to that lineage. No ProtoRes code is included here.

## SMPL body model (NOT included)

- <https://smpl.is.tue.mpg.de> — Max Planck Institute for Intelligent Systems.
- The three `basicModel_*_lbs_10_207_0_v1.0.0.pkl` files and the
  `J_regressor_*.npy` files require individual registration and acceptance of
  the SMPL Model License (research-only for the free variant; commercial use
  requires a separate MPI license). They must never be committed to this
  repository. Expected filenames and SHA-256 hashes are listed in README.md.

## AMASS dataset (NOT included)

- <https://amass.is.tue.mpg.de> — Mahmood et al., *AMASS: Archive of Motion
  Capture as Surface Shapes*, ICCV 2019.
- Requires individual registration and acceptance of the AMASS license
  (academic research only; redistribution prohibited). The raw `.npz` archives,
  the rebuilt gender-augmented CSVs, and the Feather caches derived from them
  are all AMASS-derived data and must never be committed here.

## three.js (included, `demo/vendor/`)

- `three.module.js` (r160), `OrbitControls.js`, `TransformControls.js`
- Copyright 2010-2023 Three.js Authors, MIT License
  (SPDX-License-Identifier: MIT; the license header is preserved at the top of
  `demo/vendor/three.module.js`). <https://github.com/mrdoob/three.js>
- Vendored unmodified so the demo works fully offline on cluster nodes.

## PyTorch3D (dependency, NOT included)

- <https://github.com/facebookresearch/pytorch3d> — BSD-3-Clause license,
  Copyright (c) Meta Platforms, Inc. and affiliates.
- Canonical environment uses release v0.6.2; the Hopper environment uses tag
  v0.7.9 (commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`) installed with
  `PYTORCH3D_NO_EXTENSION=1` (pure-Python `pytorch3d.transforms` only).

## Other dependencies

PyTorch (BSD-style), PyTorch Lightning (Apache-2.0), Hydra (MIT),
smplx (research license, MPI), and the remaining pinned packages in
`ops/constraints.compat.txt` / the README pip lists retain their respective
upstream licenses; they are installed from public package indexes, not
redistributed.

## Trained weights

Checkpoints published at <https://huggingface.co/Tevior/neural-ik-smpl> were
trained with the upstream non-commercial code on AMASS data using the SMPL
body model. They are therefore usable for **non-commercial research only**.

## Bundled sample data (`demo/demo_data.json`)

Eight single-frame test poses (joint coordinates, betas, and model
predictions) derived from the AMASS test split, included as a de-minimis
excerpt so the offline viewer works out of the box. The AMASS license
(https://amass.is.tue.mpg.de) governs the underlying data; no motion
sequences or raw dataset files are included.
