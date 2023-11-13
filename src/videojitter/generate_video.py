import argparse
import json
import sys
import numpy as np
import ffmpeg
from videojitter import _util, _version


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Generates a jitter test video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--output-file",
        help="Write the video to the specified file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--size",
        help="Size of the output video, in ffmpeg size format",
        default="hd1080",
    )
    argument_parser.add_argument(
        "--padding-square-width-pixels",
        help=(
            "Width of the squares used in the padding pattern, in pixels. Ignored if"
            " --padding-fullscreen-color is set"
        ),
        default=16,
        type=int,
    )
    argument_parser.add_argument(
        "--padding-square-height-pixels",
        help=(
            "Height of the squares used in the padding pattern, in pixels. Ignored if"
            " --padding-fullscreen-color is set"
        ),
        default=16,
        type=int,
    )
    argument_parser.add_argument(
        "--padding-fullscreen-color",
        help=(
            "If specified, do not use a dynamic checker pattern for padding; instead,"
            " use a static frame filled with the specified color, in ffmpeg color"
            " syntax."
        ),
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--begin-padding",
        help=(
            "How long to display the padding pattern at the beginning of the video"
            " before the test signal, in ffmpeg time format"
        ),
        default="5",
    )
    argument_parser.add_argument(
        "--end-padding",
        help=(
            "How long to display the padding pattern at the end of the video after the"
            " test signal, in ffmpeg time format"
        ),
        default="5",
    )
    return argument_parser.parse_args()


def main():
    _version.print_banner("generate_video")
    args = _parse_arguments()
    with open(args.spec_file, encoding="utf-8") as spec_file:
        spec = json.load(spec_file)

    rate = f"{spec['fps']['num']}/{spec['fps']['den']}"

    def color_input(color):
        return ffmpeg.input(f"color=c={color}:s={args.size}:r={rate}", format="lavfi")

    padding_fullscreen_color = getattr(args, "padding_fullscreen_color", None)
    padding = (
        (
            ffmpeg.filter(
                [color_input(color) for color in ["black", "white"]],
                "blend",
                all_expr=(
                    f"if(eq(gte(mod(X, {args.padding_square_width_pixels*2}),"
                    f" {args.padding_square_width_pixels}), gte(mod(Y,"
                    f" {args.padding_square_height_pixels*2}),"
                    f" {args.padding_square_height_pixels})), A, B)"
                ),
            ).filter("negate", enable="eq(mod(n, 2), 1)")
            # The loop is not strictly necessary, but makes the pipeline vastly faster
            # by ensuring both frames are only computed once and then reused.
            .filter("loop", -1, size=2, start=0)
        )
        if padding_fullscreen_color is None
        else color_input(padding_fullscreen_color)
    ).filter_multi_output("split")

    ffmpeg_spec = ffmpeg.output(
        ffmpeg.concat(
            padding[0].trim(end=args.begin_padding),
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="gray",
                s="1x1",
                r=rate,
            ).filter("scale", s=args.size),
            padding[1].trim(end=args.end_padding),
        ),
        # Include a dummy audio track as it makes the test video more realistic. Some
        # video players (especially PC software) rely on the audio track for clocking,
        # and will behave very differently if it's not there.
        ffmpeg.input("anoisesrc=c=pink:r=48000:a=0.001", format="lavfi"),
        args.output_file,
        **{
            "shortest": None,
            "profile:v": "baseline",
            "preset": "ultrafast",
            # Make the video behave like typical HD video for compatibility and to
            # ensure the video players behave similarly to a "real" video.
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
            _util.generate_frames(
                spec["transition_count"], spec["delayed_transitions"]
            ).astype(np.uint8)
            * 0xFF
        ).tobytes()
    )
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
