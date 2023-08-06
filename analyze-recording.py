#!/usr/bin/env python3

import argparse
import numpy as np
import json
import scipy.io
import scipy.signal
import sys


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given a spec file and recorded light waveform file, analyzes the recording and outputs the results to stdout."
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        type=argparse.FileType(),
    )
    argument_parser.add_argument(
        "--recording-file",
        help="Path to the input recording file",
        required=True,
        type=argparse.FileType(mode="rb"),
    )
    return argument_parser.parse_args()


def generate_reference_samples(fps_den, fps_num, frames, sample_rate):
    frame_numbers = (
        np.arange(len(frames) * fps_den * sample_rate // fps_num)
        * fps_num
        // (sample_rate * fps_den)
    )
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
    reference_duration_seconds = len(spec["frames"]) / nominal_fps
    print(
        f"Successfully loaded spec file describing {len(spec['frames'])} frames at {nominal_fps} FPS ({reference_duration_seconds} seconds)",
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

    assert (
        recording_duration_seconds > reference_duration_seconds
    ), f"Recording is shorter than expected - test video is {reference_duration_seconds} seconds long, but recording is only {recording_duration_seconds} seconds long"

    reference_samples = generate_reference_samples(
        spec["fps"]["den"], spec["fps"]["num"], spec["frames"], recording_sample_rate
    )

    cross_correlation = scipy.signal.correlate(
        recording_samples, reference_samples, mode="valid"
    )
    cross_correlation = cross_correlation / np.max(np.abs(cross_correlation))

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


analyze_recording()
