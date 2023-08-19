#!/usr/bin/env python3

import argparse
import json
import ffmpeg
import sys


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Generates a jitter test video from a spec file passed in stdin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--output-file",
        help="Path to the output video file",
        required=True,
        default=argparse.SUPPRESS,
    )
    return argument_parser.parse_args()


def generate_video():
    args = parse_arguments()
    spec = json.load(sys.stdin)
    ffmpeg_process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="gray",
            s="1x1",
            r=f"{spec['fps']['num']}/{spec['fps']['den']}",
        )
        .filter(
            "tpad",
            start_duration=5,
            stop_duration=5,
            start_mode="clone",
            stop_mode="clone",
        )
        # TODO: make the output look more similar to a typical video (i.e. typical
        # resolution, color space, audio etc)
        .output(args.output_file)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame_is_white in spec["frames"]:
        ffmpeg_process.stdin.write(b"\xff" if frame_is_white else b"\x00")
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


generate_video()
