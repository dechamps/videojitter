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
        "--sample-rate-hz",
        help="Sample rate to use for the resulting recording",
        type=float,
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
        "--invert",
        help="Invert the test signal, i.e. white is low and black is high",
        action="store_true",
    )
    return argument_parser.parse_args()


def generate_fake_recording():
    args = parse_arguments()
    sample_rate = args.sample_rate_hz
    spec = json.load(sys.stdin)

    scipy.io.wavfile.write(
        args.output_recording_file,
        sample_rate,
        np.concatenate(
            (
                -np.ones(args.begin_padding_seconds * sample_rate),
                videojitter.util.generate_fake_samples(
                    videojitter.util.generate_frames(
                        spec["transition_count"], spec["delayed_transitions"]
                    ),
                    spec["fps"]["num"],
                    spec["fps"]["den"],
                    sample_rate,
                ),
                -np.ones(args.end_padding_seconds * sample_rate),
            ),
        )
        * (-1 if args.invert else 1),
    )


generate_fake_recording()
