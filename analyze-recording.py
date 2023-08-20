#!/usr/bin/env python3

import argparse
import numpy as np
import json
import pandas as pd
import scipy.io
import scipy.signal
import sys


# TODO: decreasing this increases dispersion. Increasing also results in
# increased dispersion, *and* also makes it harder to find peaks (due to what
# looks like high frequency noise). It's not clear what causes this. It's
# possible we might be able to tighten dispersion even further if we manage to
# get an understanding of these issues.
DOWNSAMPLE_TO_FPS_TIMES = 128


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given a spec file and recorded light waveform file, analyzes the recording and outputs the results to stdout in CSV format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        type=argparse.FileType(),
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--recording-file",
        help="Path to the input recording file",
        required=True,
        type=argparse.FileType(mode="rb"),
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--downsampling-ratio",
        help=f"Downsampling ratio for preprocessing. Downsampling makes cross-correlation faster and reduces the likelihood that the analyzer will choke on high-frequency noise. (default: downsample down to just above {DOWNSAMPLE_TO_FPS_TIMES}x video FPS)",
        type=int,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--timing-minimum-sample-rate-hz",
        help="What rate to upsample the recording to (at least) before estimating frame transition timestamps. Frame transition timestamps have to land on a sample boundary, so higher sample rates make the timestamp more accurate, at the price of making analysis slower.",
        type=float,
        default=100000,
    )
    argument_parser.add_argument(
        "--black-threshold-ratio",
        help="The recording slope level below which a transition to black is deemed to have occurred, relative to the overall maximum negative slope.",
        type=float,
        default=0.8,
    )
    argument_parser.add_argument(
        "--white-threshold-ratio",
        help="The recording slope level above which a transition to white is deemed to have occurred, relative to the overall maximum positive slope.",
        type=float,
        default=0.8,
    )
    argument_parser.add_argument(
        "--output-downsampled-recording-file",
        help="(Only useful for debugging) Write the downsampled recording as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-reference-signal-file",
        help="(Only useful for debugging) Write the reference signal as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-cross-correlation-file",
        help="(Only useful for debugging) Write the cross-correlation of the recording against the reference signal as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-post-processed-recording-file",
        help="(Only useful for debugging) Write the post-processed (trimmed and possibly inverted) recording as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-recording-slope-file",
        help="(Only useful for debugging) Write the recording slope as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-upsampled-recording-slope-file",
        help="(Only useful for debugging) Write the upsampled recording slope as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-frame-transitions-file",
        help="(Only useful for debugging) Write the estimated frame transitions as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    return argument_parser.parse_args()


def generate_reference_samples(fps_den, fps_num, frames, sample_rate):
    frame_numbers = (
        np.arange(np.ceil(len(frames) * fps_den * sample_rate / fps_num))
        * fps_num
        / (sample_rate * fps_den)
    ).astype(int)
    return (np.array(frames) * 2 - 1)[frame_numbers]


def find_recording_offset(cross_correlation):
    minimum_index = np.argmin(cross_correlation)
    maximum_index = np.argmax(cross_correlation)
    inverted = abs(cross_correlation[minimum_index]) > abs(
        cross_correlation[maximum_index]
    )
    return minimum_index if inverted else maximum_index, inverted


def check_bogus_offset(cross_correlation, offset, sample_rate, fps):
    """Checks if the computed offset may be bogus.

    The offset is considered suspicious if there is an alternative candidate
    offset that is nearly as good as the one we have, and is located more than 1
    frame duration away.

    This is mostly arbitrary and might need revisiting based on how recordings
    typically look like in the field."""
    frame_duration_in_samples = sample_rate / fps
    cross_correlation_remainder = cross_correlation.copy()
    cross_correlation_remainder[
        int(offset - frame_duration_in_samples) : int(
            offset + frame_duration_in_samples
        )
    ] = 0
    best_alternative_offset = np.argmax(cross_correlation_remainder)
    if cross_correlation_remainder[best_alternative_offset] > 0.75:
        print(
            f"WARNING: detected offset may be bogus, as there is another candidate at {best_alternative_offset} ({best_alternative_offset / sample_rate} seconds). This suggests the recording may be corrupted, or the video that was played was generated from a different spec, or was playing so badly that the original frame sequence is unrecognizable. Expect garbage results.",
            file=sys.stderr,
        )


def analyze_recording():
    args = parse_arguments()
    spec = json.load(args.spec_file)
    nominal_fps = spec["fps"]["num"] / spec["fps"]["den"]
    frames = spec["frames"]
    reference_duration_seconds = len(frames) / nominal_fps
    print(
        f"Successfully loaded spec file describing {len(frames)} frames at {nominal_fps} FPS ({reference_duration_seconds} seconds)",
        file=sys.stderr,
    )

    recording_sample_rate, recording_samples = scipy.io.wavfile.read(
        args.recording_file
    )
    recording_duration_seconds = recording_samples.size / recording_sample_rate
    print(
        f"Successfully loaded recording containing {recording_samples.size} samples at {recording_sample_rate} Hz ({recording_duration_seconds} seconds)",
        file=sys.stderr,
    )

    def maybe_write_wavfile(file, samples):
        if not file:
            return
        scipy.io.wavfile.write(
            file, int(recording_sample_rate), samples.astype(np.float32)
        )

    assert (
        recording_duration_seconds > reference_duration_seconds
    ), f"Recording is shorter than expected - test video is {reference_duration_seconds} seconds long, but recording is only {recording_duration_seconds} seconds long"

    downsampling_ratio = getattr(
        args,
        "downsampling_ratio",
        np.floor(recording_sample_rate / (nominal_fps * DOWNSAMPLE_TO_FPS_TIMES)),
    )
    recording_sample_rate /= downsampling_ratio
    print(
        f"Downsampling recording by {downsampling_ratio}x (to {recording_sample_rate} Hz)",
        file=sys.stderr,
    )
    recording_samples = scipy.signal.resample_poly(
        recording_samples, up=1, down=downsampling_ratio
    )
    maybe_write_wavfile(args.output_downsampled_recording_file, recording_samples)

    reference_samples = generate_reference_samples(
        spec["fps"]["den"], spec["fps"]["num"], frames, recording_sample_rate
    )
    maybe_write_wavfile(args.output_reference_signal_file, reference_samples)

    cross_correlation = scipy.signal.correlate(
        recording_samples, reference_samples, mode="valid"
    )
    cross_correlation = cross_correlation / np.max(np.abs(cross_correlation))
    maybe_write_wavfile(args.output_cross_correlation_file, cross_correlation)

    recording_offset, inverted = find_recording_offset(cross_correlation)

    print(
        f"Test signal appears to start near sample {recording_offset} ({recording_offset / recording_sample_rate} seconds) in the recording",
        file=sys.stderr,
    )
    if inverted:
        print(
            "NOTE: recording polarity appears to be reversed (white is low and black is high). Nothing to worry about, this can be totally expected depending on recording setup. Inverting it back to compensate.",
            file=sys.stderr,
        )
        recording_samples = np.negative(recording_samples)

    check_bogus_offset(
        cross_correlation, recording_offset, recording_sample_rate, nominal_fps
    )

    recording_samples = recording_samples[
        recording_offset : recording_offset + reference_samples.size
    ]
    maybe_write_wavfile(args.output_post_processed_recording_file, recording_samples)

    recording_slope = np.diff(recording_samples)
    maybe_write_wavfile(args.output_recording_slope_file, recording_slope)

    recording_slope_approx_min = np.quantile(recording_slope, 0.01)
    recording_slope_approx_max = np.quantile(recording_slope, 0.99)
    print(
        f"Approximate recording slope range: [{recording_slope_approx_min}, {recording_slope_approx_max}]",
        file=sys.stderr,
    )
    recording_slope_black_threshold = (
        recording_slope_approx_min * args.black_threshold_ratio
    )
    recording_slope_white_threshold = (
        recording_slope_approx_max * args.white_threshold_ratio
    )
    print(
        f"Assuming that video is transitioning to black when recording slope dips below {recording_slope_black_threshold} and to white above {recording_slope_white_threshold}",
        file=sys.stderr,
    )

    upsampling_ratio = np.ceil(
        args.timing_minimum_sample_rate_hz / recording_sample_rate
    )
    recording_sample_rate *= upsampling_ratio
    recording_offset *= upsampling_ratio
    print(
        f"Upsampling recording slope by {upsampling_ratio}x to {recording_sample_rate} Hz",
        file=sys.stderr,
    )
    recording_slope = scipy.signal.resample_poly(
        recording_slope, up=upsampling_ratio, down=1
    )
    maybe_write_wavfile(args.output_upsampled_recording_slope_file, recording_slope)

    transitions_to_white = scipy.signal.find_peaks(
        recording_slope,
        height=recording_slope_white_threshold,
    )[0]
    transitions_to_black = scipy.signal.find_peaks(
        -recording_slope,
        height=-recording_slope_black_threshold,
    )[0]

    if args.output_frame_transitions_file:
        frame_transitions = np.zeros(recording_slope.size)
        frame_transitions[transitions_to_white] = 1
        frame_transitions[transitions_to_black] = -1
        maybe_write_wavfile(args.output_frame_transitions_file, frame_transitions)

    pd.Series(
        np.concatenate(
            [
                np.repeat("WHITE", transitions_to_white.size),
                np.repeat("BLACK", transitions_to_black.size),
            ]
        ),
        index=(
            np.concatenate([transitions_to_white, transitions_to_black])
            + recording_offset
        )
        / recording_sample_rate,
        name="frame",
    ).sort_index().rename_axis("recording_timestamp_seconds").to_csv(sys.stdout)


analyze_recording()
