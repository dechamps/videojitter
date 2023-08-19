#!/usr/bin/env python3

import argparse
import json
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
        "--repeat-probability",
        help="Probability that a frame will be repeated",
        default=0.5,
        type=float,
    )
    return argument_parser.parse_args()


def generate_frames(count, repeat_probability):
    """Generate n frames alternating between black and white.

    Some frames are repeated at random; however a given frame will never appear
    more than twice in a row, and the first and last frames can never be part
    of a repeat."""
    last_frames = []
    for frame_index in range(0, count):
        if not last_frames:
            frame = random.random() > 0.5
        else:
            frame = last_frames[-1]
            if (
                len(last_frames) < 2
                or last_frames[0] == last_frames[1]
                or frame_index == count - 1
                or random.random() > repeat_probability
            ):
                frame = not frame
        yield frame
        last_frames = last_frames[0 if len(last_frames) < 2 else -1 :] + [frame]


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
            "frames": list(generate_frames(frame_count, args.repeat_probability)),
        },
        sys.stdout,
    )
    print()


generate_spec()
