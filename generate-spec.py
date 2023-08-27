#!/usr/bin/env python3

import argparse
import json
import numpy as np
import random
import sys


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Generates a spec file for video jitter testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--fps-num", help="FPS fraction numerator", default=24000, type=int
    )
    argument_parser.add_argument(
        "--fps-den", help="FPS fraction denominator", default=1001, type=int
    )
    argument_parser.add_argument(
        "--duration-seconds", help="Test duration in seconds", default=60, type=float
    )
    argument_parser.add_argument(
        "--frame-repeat-ratio",
        help="Ratio of transitions that will trigger a frame repeat",
        default=0.1,
        type=float,
    )
    return argument_parser.parse_args()


def generate_frames(count, frame_repeat_ratio, rng):
    """Generate n frames alternating between black and white.

    Some frames are repeated at random; however a given frame will never appear
    more than twice in a row, and the first and last frames can never be part
    of a repeat."""
    transition_count = int(np.ceil(count / (1 + frame_repeat_ratio)))
    transition_frame_counts = np.ones(transition_count, dtype=int)
    transition_frame_counts[
        rng.choice(
            transition_frame_counts.size - 2, count - transition_count, replace=False
        )
        + 1
    ] = 2
    return (
        ((np.arange(transition_frame_counts.size) + rng.choice([0, 1])) % 2)
        .astype(bool)
        .repeat(transition_frame_counts)
    )


def generate_spec():
    args = parse_arguments()
    frame_count = int(args.duration_seconds * (args.fps_num / args.fps_den))
    print(
        f"Generating {frame_count} frames at {args.fps_num / args.fps_den} FPS",
        file=sys.stderr,
    )

    json.dump(
        {
            "fps": {"num": args.fps_num, "den": args.fps_den},
            "frames": generate_frames(
                frame_count, args.frame_repeat_ratio, np.random.default_rng()
            ).tolist(),
        },
        sys.stdout,
    )
    print()


generate_spec()
