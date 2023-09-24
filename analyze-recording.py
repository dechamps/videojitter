#!/usr/bin/env python3

import argparse
import numpy as np
import json
import pandas as pd
import scipy.io
import scipy.signal
import sys
import videojitter.util


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
        "--min-edge-separation-seconds",
        help="The approximate minimum time interval between edges that the analyzer is expected to resolve. If the interval between two edges is shorter than this, the analyzer might become unable to precisely estimate timing differences between the two edges; if the interval is half as short, both edges may be missed entirely. On the other hand, setting this lower lets more high frequency noise in, possibly causing edge detection false positives.",
        type=float,
        default=0.001,
    )
    argument_parser.add_argument(
        "--max-edge-spread-seconds",
        help='The approximate maximum amount of time an edge can spread over before the analyzer might fail to detect it. This determines how "sluggish" the system response (including the instrument) is allowed to be. Setting this higher will allow slower transitions to be detected, but lets more low frequency noise in, possibly causing faster transitions to be missed due to drift.',
        type=float,
        default=0.020,
    )
    argument_parser.add_argument(
        "--min-edges-ratio",
        help="The minimum number of edges that can be assumed to be present in the test signal, as a ratio of the number of transitions implied by the spec. Used in combination with --edge-slope-threshold.",
        type=float,
        default=0.25,
    )
    argument_parser.add_argument(
        "--edge-slope-threshold",
        help="The absolute slope-at-zero-crossing threshold above which an edge will be recorded, as a ratio of the Nth steepest slope, where N is dictated by --min-edges-ratio. Determines how sensitive the analyzer is when detecting edges.",
        type=float,
        default=0.3,
    )
    argument_parser.add_argument(
        "--boundaries-signal-frames",
        help="The length of the reference signal used to detect the beginning and end of the test signal within the recording, in nominal frame durations.",
        type=int,
        default=11,
    )
    argument_parser.add_argument(
        "--boundaries-score-threshold-ratio",
        help="How well does a given portion of the recording have to match the reference sequence in order for it to be considered as the beginning or end of the test signal, as a ratio of the best match anywhere in the recording.",
        type=float,
        default=0.5,
    )
    argument_parser.add_argument(
        "--output-downsampled-file",
        help="(Only useful for debugging) Write the downsampled recording as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-highpassed-file",
        help="(Only useful for debugging) Write the highpassed recording as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-boundaries-signal-file",
        help="(Only useful for debugging) Write the boundaries reference signal as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-cross-correlation-file",
        help="(Only useful for debugging) Write the cross-correlation of recording against the boundaries reference signal as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-boundary-candidates-file",
        help="(Only useful for debugging) Write the boundary candidates as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-trimmed-file",
        help="(Only useful for debugging) Write the trimmed recording as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-zero-crossing-slopes-file",
        help="(Only useful for debugging) Write the slopes at zero crossings as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    argument_parser.add_argument(
        "--output-edges-file",
        help="(Only useful for debugging) Write the estimated edges as a WAV file to the given path",
        type=argparse.FileType(mode="wb"),
    )
    return argument_parser.parse_args()


def generate_boundaries_reference_samples(frame_count, fps_num, fps_den, sample_rate):
    return videojitter.util.generate_fake_samples(
        np.tile([False, True], int(np.ceil(frame_count / 2)))[0:frame_count],
        fps_num,
        fps_den,
        sample_rate,
    )


def analyze_recording():
    args = parse_arguments()
    assert (
        args.boundaries_signal_frames % 2 != 0
    ), "The number of frames in the boundaries reference signal should be odd so that the signal begins and ends on the same frame"
    spec = json.load(args.spec_file)
    nominal_fps = spec["fps"]["num"] / spec["fps"]["den"]
    expected_transition_count = spec["transition_count"]
    frames = videojitter.util.generate_frames(
        expected_transition_count, spec["delayed_transitions"]
    )
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

    def format_index(index):
        return f"sample {index} ({index / recording_sample_rate} seconds)"

    def maybe_write_wavfile(file, samples, normalize=False):
        if not file:
            return
        if normalize:
            samples = samples / np.max(np.abs(samples))
        scipy.io.wavfile.write(
            file, int(recording_sample_rate), samples.astype(np.float32)
        )

    # If we assume the worst-case scenario where all the edges in the input
    # signal are separated by the min threshold Tm, then the fundamental
    # period in the input signal is 2*Tm (because the period is a full rising
    # edge + falling edge cycle), and the fundamental frequency is 1/(2*Tm).
    # However we also need to preserve the second harmonic, because that carries
    # the information about waveform asymmetry - if we lose that, then it will
    # look like every rising edge occurs exactly halfway between every falling
    # edge (and vice-versa), thus destroying important timing information that
    # the user may care about (consider e.g. 3:2 patterns). Therefore we need to
    # preserve frequencies up to 1/Tm, and thus, per Nyquist, we need a sampling
    # rate of at least 2/Tm.
    downsampling_ratio = np.floor(
        0.5 * args.min_edge_separation_seconds * recording_sample_rate
    )
    recording_sample_rate /= downsampling_ratio
    print(
        f"Downsampling recording by {downsampling_ratio}x (to {recording_sample_rate} Hz)",
        file=sys.stderr,
    )
    recording_samples = scipy.signal.resample_poly(
        recording_samples,
        up=1,
        down=downsampling_ratio,
    )
    maybe_write_wavfile(args.output_downsampled_file, recording_samples)

    # If we asssume the worst-case scenario where the input signal is a
    # perfectly smooth (in other words, perfectly sluggish) sinusoid where each
    # edge is separated by the max threshold, Tm, then the only period in the
    # input signal is the fundamental period which is 2*Tm (because a period is
    # a rising edge followed by a falling edge), and the fundamental frequency
    # is 1/(2*Tm), which is the minimum frequency we need to preserve.
    min_frequency = 0.5 / args.max_edge_spread_seconds
    print(
        f"Removing frequencies lower than {min_frequency} Hz from the recording",
        file=sys.stderr,
    )
    # Note: the default filter impulse response rings in the time domain,
    # resulting in spurious zero crossings that weren't there in the original
    # signal. This is probably fine though, because it's ringing at the cutoff
    # frequency, which is low enough that it wouldn't produce a steep zero
    # crossing that could be mistaken for an edge. If we really want to get rid
    # of these, one idea could be to use a Gaussian or raised cosine filter
    # instead.
    recording_samples = scipy.signal.convolve(
        recording_samples,
        scipy.signal.firwin(
            int(np.ceil(10 * recording_sample_rate / min_frequency / 2)) * 2 + 1,
            min_frequency,
            pass_zero=False,
            fs=recording_sample_rate,
        ),
        "same",
    )
    maybe_write_wavfile(args.output_highpassed_file, recording_samples)

    boundaries_reference_samples = generate_boundaries_reference_samples(
        args.boundaries_signal_frames,
        spec["fps"]["num"],
        spec["fps"]["den"],
        recording_sample_rate,
    )
    maybe_write_wavfile(
        args.output_boundaries_signal_file, boundaries_reference_samples
    )

    cross_correlation = scipy.signal.correlate(
        recording_samples, boundaries_reference_samples, mode="valid"
    )
    maybe_write_wavfile(
        args.output_cross_correlation_file,
        cross_correlation,
        normalize=True,
    )

    abs_cross_correlation = np.abs(cross_correlation)
    boundary_candidates = (
        abs_cross_correlation
        >= np.max(abs_cross_correlation) * args.boundaries_score_threshold_ratio
    )
    maybe_write_wavfile(
        args.output_boundary_candidates_file,
        boundary_candidates,
    )

    boundary_candidate_indexes = np.nonzero(boundary_candidates)[0]
    assert boundary_candidate_indexes.size > 1
    test_signal_start_index = boundary_candidate_indexes[0]
    test_signal_end_index = (
        boundary_candidate_indexes[-1] + boundaries_reference_samples.size
    )
    print(
        f"Test signal appears to start at {format_index(test_signal_start_index)} and end at {format_index(test_signal_end_index)} in the recording.",
        file=sys.stderr,
    )

    recording_samples = recording_samples[test_signal_start_index:test_signal_end_index]
    maybe_write_wavfile(args.output_trimmed_file, recording_samples)

    zero_crossing_indexes = np.nonzero(np.diff(recording_samples > 0))[0]
    recording_slope = np.diff(recording_samples)
    zero_crossing_slopes = recording_slope[zero_crossing_indexes]
    if args.output_zero_crossing_slopes_file:
        recording_zero_crossing_slopes = np.zeros(recording_samples.size)
        recording_zero_crossing_slopes[zero_crossing_indexes] = zero_crossing_slopes
        maybe_write_wavfile(
            args.output_zero_crossing_slopes_file, recording_zero_crossing_slopes
        )
    zero_crossing_absolute_slopes = np.abs(zero_crossing_slopes)

    # Noise in the recording signal can result in lots of spurious zero
    # crossings, especially if the last frame transition happened some time ago
    # and the signal is now back to hovering around zero. We need a way to tell
    # the spurious zero crossings apart from the true edges. The solution we use
    # here relies on the signal slope at the zero crossing as the criterion. We
    # need to decide on a slope threshold below which a zero crossing will be
    # rejected. The correct threshold must be below the minimum true edge slope,
    # which depends on overall signal shape and amplitude, so we'll have to take
    # a guess. We could reference our guess on the steepest slope among all zero
    # crossings in the signal - that's pretty much guaranteed to be a valid
    # edge - but that would make the threshold very sensitive to an isolated
    # outlier. To avoid this problem we base our guess on the Nth steepest
    # slope, where N is large enough to mitigate the influence of outliers, but
    # small enough that we can be reasonably confident we're still going to pick
    # a valid edge even if the test signal contains fewer edges than expected.
    # We then multiply that reference with a fudge factor to allow for edges
    # with slightly smaller slopes, and that's our threshold.
    zero_crossing_partition_nth = int(
        np.round(expected_transition_count * args.min_edges_ratio)
    )
    edge_slope_reference = np.partition(
        zero_crossing_absolute_slopes, -zero_crossing_partition_nth
    )[-zero_crossing_partition_nth]
    edge_slope_threshold = edge_slope_reference * args.edge_slope_threshold
    valid_edge = zero_crossing_absolute_slopes > edge_slope_threshold
    valid_edge_indexes = zero_crossing_indexes[valid_edge]
    print(
        f"{zero_crossing_partition_nth}nth steepest absolute zero crossing slope is {edge_slope_reference}. Kept {valid_edge_indexes.size} edges (out of {zero_crossing_indexes.size} zero crossings) whose absolute slope is above {edge_slope_threshold}. First edge is right after {format_index(valid_edge_indexes[0])} and last edge is right after {format_index(valid_edge_indexes[-1])}.",
        file=sys.stderr,
    )
    assert valid_edge_indexes.size > 0
    valid_edge_is_rising = zero_crossing_slopes[valid_edge] > 0
    if args.output_edges_file:
        recording_edges = np.zeros(recording_samples.size)
        recording_edges[valid_edge_indexes] = valid_edge_is_rising * 2 - 1
        maybe_write_wavfile(args.output_edges_file, recording_edges)

    # `valid_edge_indexes` refers to the sample right before the zero crossing.
    # This is an integer index whose precision is inherently limited by the
    # sample rate. To improve the precision, we use linear interpolation
    # between that sample and the next to compute a better estimate of the true
    # position of the zero crossing.
    # TODO: while this approach does improve precision tremendously, we still
    # observe improvements as sample rate increases, suggesting that we still
    # haven't reached the true potential of the underlying signal. This is
    # unsurprising given the literature (e.g. Svitlov, Sergiy et al. “Accuracy
    # assessment of the two-sample zero-crossing detection in a sinusoidal
    # signal.” Metrologia 49 (2012): 413 - 424). We could either upsample first,
    # or we could try more sophisticated interpolation techniques such as cubic
    # or splines.
    valid_edge_positions = (
        valid_edge_indexes
        - recording_samples[valid_edge_indexes] / recording_slope[valid_edge_indexes]
    )

    edges = pd.Series(
        valid_edge_is_rising,
        index=pd.Index(
            (valid_edge_positions + test_signal_start_index) / recording_sample_rate,
            name="recording_timestamp_seconds",
        ),
        name="frame",
    )

    first_edge = edges.iloc[0]
    last_edge = edges.iloc[-1]
    if first_edge == last_edge:
        print(
            f"WARNING: the first and last edges are both {'rising' if first_edge else 'falling'}. This doesn't make sense as the first and last frames of the test video are supposed to be both black. Unable to determine transition directions as a result.",
            file=sys.stderr,
        )
    else:
        print(
            f"First edge is {'rising' if first_edge else 'falling'} and last edge is {'rising' if last_edge else 'falling'}. Deducing that a falling edge means a transition to {'black' if first_edge else 'white'} and a rising edge means a transition to {'white' if first_edge else 'black'}.",
            file=sys.stderr,
        )
        if not first_edge:
            edges = ~edges

    edges.to_csv(sys.stdout)


analyze_recording()
