#!/usr/bin/env python3

import argparse
import sys
import json
import numpy as np
import scipy.io
import videojitter.util


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given a spec file passed in stdin, generates a recording faking what a real instrument would output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--output-recording-file",
        help="Path to the resulting WAV file",
        required=True,
        type=argparse.FileType(mode="wb"),
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--internal-sample-rate-hz",
        help="The (minimum) internal sample rate used for generating the signal. Directly determines the time resolution of the frame transitions.",
        type=int,
        default=100000,
    )
    argument_parser.add_argument(
        "--output-sample-rate-hz",
        help="Sample rate to resample to before writing the recording.",
        type=int,
        default=48000,
    )
    argument_parser.add_argument(
        "--begin-padding-seconds",
        help="Duration of the padding before the test signal",
        type=float,
        default=5,
    )
    argument_parser.add_argument(
        "--end-padding-seconds",
        help="Duration of the padding after the test signal",
        type=float,
        default=5,
    )
    argument_parser.add_argument(
        "--white-duration-overshoot",
        help="Make white frames overshoot into the next frame by this amount of time, relative to the nominal frame duration. Can be used to simulate asymmetry.",
        type=float,
        default=0.05,
    )
    argument_parser.add_argument(
        "--even-duration-overshoot",
        help="Make even frames overshoot into odd frames by this amount of time, relative to the nominal frame duration. Set to 0.2 (or -0.2) to simulate a 3:2 (or 2:3) 24p@60Hz-like pattern.",
        type=float,
        default=0,
    )
    argument_parser.add_argument(
        "--invert",
        help="Invert the test signal, i.e. white is low and black is high",
        action="store_true",
    )
    argument_parser.add_argument(
        "--amplitude",
        help="Amplitude of the resulting signal, where 1.0 is full scale.",
        type=float,
        default=0.8,
    )
    return argument_parser.parse_args()


def generate_fake_recording():
    args = parse_arguments()
    sample_rate = args.internal_sample_rate_hz
    spec = json.load(sys.stdin)

    assert args.internal_sample_rate_hz > args.output_sample_rate_hz
    downsample_ratio = int(
        np.ceil(args.internal_sample_rate_hz / args.output_sample_rate_hz)
    )
    sample_rate = args.output_sample_rate_hz * downsample_ratio
    print(f"Using internal sample rate of {sample_rate} Hz", file=sys.stderr)

    scipy.io.wavfile.write(
        args.output_recording_file,
        args.output_sample_rate_hz,
        scipy.signal.resample_poly(
            np.concatenate(
                (
                    -np.ones(int(np.round(args.begin_padding_seconds * sample_rate))),
                    videojitter.util.generate_fake_samples(
                        videojitter.util.generate_frames(
                            spec["transition_count"], spec["delayed_transitions"]
                        ),
                        spec["fps"]["num"],
                        spec["fps"]["den"],
                        sample_rate,
                        white_duration_overshoot=args.white_duration_overshoot,
                        even_duration_overshoot=args.even_duration_overshoot,
                    ),
                    -np.ones(int(np.round(args.end_padding_seconds * sample_rate))),
                ),
            )
            * ((-1 if args.invert else 1) * args.amplitude),
            up=1,
            down=downsample_ratio,
        ),
    )


generate_fake_recording()
