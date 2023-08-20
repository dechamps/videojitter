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
    argument_parser.add_argument(
        "--begin-padding",
        help="How long to hold the first frame at the beginning of the video, in ffmpeg time format",
        default="5",
    )
    argument_parser.add_argument(
        "--end-padding",
        help="How long to hold the last frame at the end of the video, in ffmpeg time format",
        default="5",
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
            start_duration=args.begin_padding,
            stop_duration=args.end_padding,
            start_mode="clone",
            stop_mode="clone",
        )
        # TODO: add audio (important for video player clocking)
        .output(
            args.output_file,
            **{
                "s": "hd1080",
                "pix_fmt": "yuv420p",
                "profile:v": "baseline",
                "preset": "ultrafast",
                "color_primaries": "bt709",
                "color_trc": "bt709",
                "colorspace": "bt709",
                "color_range": "tv",
            },
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame_is_white in spec["frames"]:
        ffmpeg_process.stdin.write(b"\xff" if frame_is_white else b"\x00")
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


generate_video()
