#!/usr/bin/env python3
"""Render consecutive processed AMASS frames for visual data QA."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from smplik.smpl.smpl_info import SMPL_JOINT_NAMES


PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def position_columns():
    return [
        f"BonePositions_{joint}_{axis}"
        for joint in SMPL_JOINT_NAMES[:24]
        for axis in ("X", "Y", "Z")
    ]


def set_axes(ax, limit):
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=18, azim=-68)
    ax.set_box_aspect((1, 1, 1))


def draw_pose(ax, pose, limit, title):
    set_axes(ax, limit)
    for child, parent in enumerate(PARENTS):
        if parent < 0:
            continue
        segment = pose[[parent, child]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="tab:blue", linewidth=2)
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color="tab:orange", s=12)
    ax.set_title(title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "datasets/amass_gender_augment_cache_v1/"
            "BioMotionLab_NTroje/BioMotionLab_NTroje.csv"
        ),
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("run_reports/visual_qa")
    )
    args = parser.parse_args()

    columns = position_columns()
    dataframe = pd.read_csv(
        args.csv, usecols=columns + ["Gender_V"], nrows=args.frames
    )
    poses = dataframe[columns].to_numpy(dtype=np.float32).reshape(-1, 24, 3)
    poses = poses - poses[:, :1, :]
    gender = int(dataframe["Gender_V"].iloc[0])
    limit = float(np.max(np.abs(poses))) * 1.1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_path = args.output_dir / "amass_processed_motion_contact_sheet.png"
    gif_path = args.output_dir / "amass_processed_motion.gif"

    selected = np.linspace(0, len(poses) - 1, 12, dtype=int)
    figure = plt.figure(figsize=(16, 12))
    for panel, frame_index in enumerate(selected, start=1):
        axis = figure.add_subplot(3, 4, panel, projection="3d")
        draw_pose(axis, poses[frame_index], limit, f"frame {frame_index}")
    figure.suptitle(f"Processed AMASS consecutive frames, gender={gender}")
    figure.tight_layout()
    figure.savefig(contact_path, dpi=140)
    plt.close(figure)

    animation_figure = plt.figure(figsize=(6, 6))
    animation_axis = animation_figure.add_subplot(111, projection="3d")

    def update(frame_index):
        animation_axis.cla()
        draw_pose(
            animation_axis,
            poses[frame_index],
            limit,
            f"Processed AMASS frame {frame_index}, gender={gender}",
        )

    animation = FuncAnimation(
        animation_figure, update, frames=range(0, len(poses), 2), interval=80
    )
    animation.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(animation_figure)

    print(f"Contact sheet: {contact_path.resolve()}")
    print(f"Motion GIF: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
