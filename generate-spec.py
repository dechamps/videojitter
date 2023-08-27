#!/usr/bin/env python3

import argparse
import json
import numpy as np
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
        help="Ratio of transitions that will have a repeated frame before or after the transition",
        default=0.1,
        type=float,
    )
    return argument_parser.parse_args()


def nonconsecutive_integers_count(minimum, maximum):
    """Returns the number of non-consecutive integers between `minimum` and
    `maximum`, inclusive."""
    return


def nonconsecutive_random_integers(minimum, maximum, count):
    """Generates an array of `count` integers between `minimum` and `maximum`.

    The bounds are inclusive.

    The distance between any two numbers in the output is guaranteed to be at
    least 2 (e.g. the output can include 42 or 43, but not both).

    `count` is clamped to the number of nonconsecutive integers that can fit
    between `minimum` and `maximum`.
    """
    # Inspired by https://stackoverflow.com/a/31060350/172594
    count = min(count, int((maximum - minimum + 2) / 2))
    return (
        np.sort(np.random.choice(maximum - minimum - count + 2, count, replace=False))
        + np.arange(count)
        + minimum
    )


def generate_frames(count, frame_repeat_ratio, rng):
    """Generate n frames alternating between black and white.

    Some frames are repeated at random, constrained by the following rules:
     - A given frame will never appear more than twice in a row;
     - Frames cannot be repeated on *both* sides of a transition (i.e. no
       consecutive repeats);
     - The first and last frames can never be repeated."""
    transition_frame_counts = np.ones(
        int(np.ceil(count / (1 + frame_repeat_ratio))), dtype=int
    )
    transition_frame_counts[
        nonconsecutive_random_integers(
            1,
            transition_frame_counts.size - 2,
            int(np.round(transition_frame_counts.size * frame_repeat_ratio)),
        )
    ] = 2
    return (
        ((np.arange(transition_frame_counts.size) + rng.choice([0, 1])) % 2)
        .astype(bool)
        .repeat(transition_frame_counts)
    )


def generate_spec():
    args = parse_arguments()
    # Frame repeat ratio cannot be higher than 50% as we don't allow consecutive
    # repeats.
    assert (
        args.frame_repeat_ratio <= 0.5
    ), "Frame repeat ratio cannot be higher than 0.5"

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
