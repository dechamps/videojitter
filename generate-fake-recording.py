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
        default=0.5,
    )
    argument_parser.add_argument(
        "--gaussian-filter-stddev-seconds",
        help="Run the signal through a gaussian filter with this specific standard deviation. Can be used to simulate the response of a typical light sensor. As an approximate rule of thumb, to simulate a light sensor that takes N seconds to reach steady state, set this option to N/2.6. Set to zero to disable.",
        type=float,
        default=0.001,
    )
    argument_parser.add_argument(
        "--high-pass-filter-hz",
        help="Run the signal through a single-pole Butterworth high-pass IIR filter with the specified cutoff frequency. Can be used to simulate an AC-coupled instrument. Set to zero to disable.",
        type=float,
        default=10,
    )
    return argument_parser.parse_args()


def apply_gaussian_filter(samples, stddev_samples):
    kernel = scipy.signal.windows.gaussian(
        M=int(np.round(stddev_samples * 10)),
        std=stddev_samples,
    )
    return scipy.signal.convolve(
        samples,
        kernel / np.sum(kernel),
        mode="same",
    )


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

    samples = scipy.signal.resample_poly(
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
            )
        ),
        up=1,
        down=downsample_ratio,
    ) * ((-1 if args.invert else 1) * args.amplitude)
    sample_rate = sample_rate / downsample_ratio

    if args.gaussian_filter_stddev_seconds:
        gaussian_filter_stddev_samples = (
            args.gaussian_filter_stddev_seconds * sample_rate
        )
        samples = apply_gaussian_filter(samples, gaussian_filter_stddev_samples)

    if args.high_pass_filter_hz:
        samples = scipy.signal.sosfilt(
            scipy.signal.butter(
                1, args.high_pass_filter_hz, "highpass", fs=sample_rate, output="sos"
            ),
            samples,
        )

    scipy.io.wavfile.write(
        args.output_recording_file,
        args.output_sample_rate_hz,
        samples,
    )


generate_fake_recording()
