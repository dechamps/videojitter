import argparse
import json
import sys


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Generates a spec file for video jitter testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--output-spec-file",
        help="Write the spec to the specified file",
        required=True,
        default=argparse.SUPPRESS,
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
        help=(
            "(Not recommended) Do not delay a single transition (i.e. repeat a single"
            " frame) the middle of the video. This ensures every single frame is"
            " accounted for and simplifies analysis, but breaks black/white offset"
            " compensation in the presence of patterns affecting pairs of consecutive"
            " frames (e.g. 3:2, 24p@60Hz)."
        ),
        action="store_true",
    )
    return argument_parser.parse_args()


def main():
    args = _parse_arguments()

    delayed_transition = not args.no_delayed_transition

    transition_count = int(args.duration_seconds * args.fps_num // args.fps_den)
    if delayed_transition:
        transition_count -= 1
    if transition_count % 2 != 0:
        # Keep the transition count even so that we begin and end with a black frame.
        transition_count += 1

    print(
        f"{transition_count} transitions at {args.fps_num / args.fps_den} FPS",
        file=sys.stderr,
    )

    with open(args.output_spec_file, "w", encoding="utf-8") as spec_file:
        json.dump(
            {
                "fps": {"num": args.fps_num, "den": args.fps_den},
                "transition_count": transition_count,
                "delayed_transitions": (
                    [transition_count // 2] if delayed_transition else []
                ),
            },
            spec_file,
        )
