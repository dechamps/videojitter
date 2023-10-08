#!/usr/bin/env python3

import argparse
import json
import numpy as np
import ffmpeg
import sys
import videojitter.util


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

    ffmpeg_spec = ffmpeg.output(
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="gray",
            s="1x1",
            r=f"{spec['fps']['num']}/{spec['fps']['den']}",
        ).filter(
            "tpad",
            start_duration=args.begin_padding,
            stop_duration=args.end_padding,
            start_mode="clone",
            stop_mode="clone",
        ),
        # Include a dummy audio track as it makes the test video more
        # realistic. Some video players (especially PC software) rely on the
        # audio track for clocking, and will behave very differently if it's
        # not there.
        ffmpeg.input("anoisesrc=c=pink:r=48000:a=0.001", format="lavfi"),
        args.output_file,
        **{
            "shortest": None,
            "profile:v": "baseline",
            "preset": "ultrafast",
            # Make the video behave like typical HD video for compatibility
            # and to ensure the video players behave similarly to a "real"
            # video.
            "s": "hd1080",
            "pix_fmt": "yuv420p",
            "color_primaries": "bt709",
            "color_trc": "bt709",
            "colorspace": "bt709",
            "color_range": "tv",
            "acodec": "ac3",
            "loglevel": "verbose",
        },
    ).overwrite_output()

    print(ffmpeg_spec.compile(), file=sys.stderr)

    ffmpeg_process = ffmpeg_spec.run_async(pipe_stdin=True)
    ffmpeg_process.stdin.write(
        (
            videojitter.util.generate_frames(
                spec["transition_count"], spec["delayed_transitions"]
            ).astype(np.uint8)
            * 0xFF
        ).tobytes()
    )
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()


generate_video()
