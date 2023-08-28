#!/usr/bin/env python3

import argparse
import math
import json
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
        "--no-delayed-transition",
        help="(Not recommended) Do not delay a single transition (i.e. repeat a single frame) the middle of the video. This ensures every single frame is accounted for and simplifies analysis, but produces misleading results in the presence of patterns affecting pairs of consecutive frames (e.g. 3:2, 24p@60Hz) due to black vs. white transition lag compensation.",
        action="store_true",
    )
    return argument_parser.parse_args()


def generate_spec():
    args = parse_arguments()

    delayed_transition = not args.no_delayed_transition

    transition_count = math.floor(args.duration_seconds * args.fps_num / args.fps_den)
    if delayed_transition:
        transition_count -= 1
    if transition_count % 2 != 0:
        # Keep the transition count even so that we begin and end with a black
        # frame.
        transition_count += 1

    print(
        f"{transition_count} transitions at {args.fps_num / args.fps_den} FPS",
        file=sys.stderr,
    )

    json.dump(
        {
            "fps": {"num": args.fps_num, "den": args.fps_den},
            "transition_count": transition_count,
            "delayed_transitions": [int(transition_count / 2)]
            if delayed_transition
            else [],
        },
        sys.stdout,
    )
    print()


generate_spec()
