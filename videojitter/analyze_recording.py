import argparse
import numpy as np
import json
import pandas as pd
import scipy.io
import scipy.signal
import sys
import videojitter.util


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given the recorded light waveform file, analyzes the recording and writes the resulting frame transition timestamps to a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--output-frame-transitions-csv-file",
        help="Write the frame transition information to the specified CSV file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--recording-file",
        help="Path to the input recording file",
        required=True,
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
        help='The approximate maximum amount of time an edge can spread over before the analyzer might fail to detect it. This determines how "sluggish" the system response (including the instrument) is allowed to be. Setting this higher will allow slower transitions to be detected, but lets more low frequency noise in, possibly causing other transitions to be missed or mistimed due to drift.',
        type=float,
        default=0.020,
    )
    argument_parser.add_argument(
        "--upsampling-ratio",
        help="How much to upsample the signal before attempting to find edges. Upsampling reduces interpolation error, thereby improving the precision of zero crossing calculations (timestamp and slope), but makes processing slower.",
        type=int,
        default=2,
    )
    argument_parser.add_argument(
        "--min-edges-ratio",
        help="The minimum number of edges that can be assumed to be present in the test signal, as a ratio of the number of transitions implied by the spec. Used in combination with --edge-slope-threshold.",
        type=float,
        default=0.6,
    )
    argument_parser.add_argument(
        "--edge-slope-threshold",
        help="The absolute slope-at-zero-crossing threshold above which an edge will be recorded, as a ratio of the Nth steepest slope, where N is dictated by --min-edges-ratio. Determines how sensitive the analyzer is when detecting edges.",
        type=float,
        default=0.5,
    )
    argument_parser.add_argument(
        "--boundary-edge-rejection-neighbor-count",
        help="The number of neighboring edges that are used to calculate the reference for boundary edge rejection. See --boundary-edge-rejection-slope-threshold.",
        type=int,
        default=20,
    )
    argument_parser.add_argument(
        "--boundary-edge-rejection-slope-threshold",
        help="The slope below which the first or last edge will be rejected, as a ratio of the minimum of the neighboring edges. See --boundary-edge-rejection-neighbor-count. This is used to reject spurious edges from the test video padding boundary.",
        type=float,
        default=0.95,
    )
    argument_parser.add_argument(
        "--boundaries-signal-periods",
        help="The length of the reference signal used to detect the beginning and end of the test signal within the recording, in periods (i.e. pairs of frames).",
        type=int,
        default=5,
    )
    argument_parser.add_argument(
        "--boundaries-score-threshold-ratio",
        help="How well does a given portion of the recording have to match the reference sequence in order for it to be considered as the beginning or end of the test signal, as a ratio of the best match anywhere in the recording.",
        type=float,
        default=0.5,
    )
    argument_parser.add_argument(
        "--output-debug-files-prefix",
        help="If set, will write a bunch of files that describe the internal state of the analyzer at various stages of the pipeline under the specified file name prefix. Interpreting this data requires some familiarity with analyzer internals.",
        default=argparse.SUPPRESS,
    )
    return argument_parser.parse_args()


def _generate_boundaries_reference_samples(period_count, fps_num, fps_den, sample_rate):
    return videojitter.util.generate_fake_samples(
        np.tile([False, True], period_count),
        fps_num,
        fps_den,
        sample_rate,
    )


def _generate_highpass_kernel(cutoff_frequency_hz, sample_rate):
    # A naive highpass filter can cause massive ringing in the time domain,
    # creating additional spurious zero crossings. To avoid ringing we choose a
    # very short FIR filter length that only lets in the main lobe of the sinc
    # function.
    #
    # TODO: one would expect this filter to be down -3 dB at the cutoff
    # frequency, but it's actually down approx. -8 dB. Investigate.
    return videojitter.util.firwin(
        int(np.ceil(sample_rate / cutoff_frequency_hz / 2)) * 2 + 1,
        cutoff_frequency_hz,
        fs=sample_rate,
        pass_zero=False,
    )


def _first_relative_to_same_sign_neighbor_mean(x, neighbor_count):
    """Out of the points in `x` that are the same sign as `x[0]`, keep the first
    `neighbor_count` points, then return `x[0]` divided by the mean of these
    points."""
    first = x[0]
    x = x[(x > 0) if (first > 0) else (x < 0)]
    return first / (np.min if first > 0 else np.max)(x[1 : neighbor_count + 1])


def main():
    args = _parse_arguments()
    with open(args.spec_file) as spec_file:
        spec = json.load(spec_file)
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

    wavfile_index = 0
    debug_files_prefix = getattr(args, "output_debug_files_prefix", None)

    def maybe_write_debug_wavfile(name, samples, normalize=False):
        if debug_files_prefix is None:
            return
        if normalize:
            samples = samples / np.max(np.abs(samples))
        nonlocal wavfile_index
        scipy.io.wavfile.write(
            f"{debug_files_prefix}{wavfile_index:02}_{name}.wav",
            int(recording_sample_rate),
            samples.astype(np.float32),
        )
        wavfile_index += 1

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
    maybe_write_debug_wavfile("downsampled", recording_samples)

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

    highpass_kernel = _generate_highpass_kernel(min_frequency, recording_sample_rate)
    maybe_write_debug_wavfile("highpass_kernel", highpass_kernel)
    recording_samples = scipy.signal.convolve(
        recording_samples, highpass_kernel, "same"
    )
    maybe_write_debug_wavfile("highpassed", recording_samples)

    boundaries_reference_samples = _generate_boundaries_reference_samples(
        args.boundaries_signal_periods,
        spec["fps"]["num"],
        spec["fps"]["den"],
        recording_sample_rate,
    )
    maybe_write_debug_wavfile("boundaries_reference", boundaries_reference_samples)

    cross_correlation = scipy.signal.correlate(
        recording_samples,
        boundaries_reference_samples / boundaries_reference_samples.size,
        mode="valid",
    )
    maybe_write_debug_wavfile(
        "cross_correlation",
        cross_correlation,
    )

    abs_cross_correlation = np.abs(cross_correlation)
    boundary_candidates = (
        abs_cross_correlation
        >= np.max(abs_cross_correlation) * args.boundaries_score_threshold_ratio
    )
    maybe_write_debug_wavfile("boundary_candidates", boundary_candidates)

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
    maybe_write_debug_wavfile("trimmed", recording_samples)

    # Upsampling improves the accuracy of our zero crossing estimates (slope,
    # position), which are based on linear interpolation. For more background
    # on how sampling rate (and other factors) affect the accuracy of
    # zero-crossing estimates, see Svitlov, Sergiy et al. “Accuracy assessment
    # of the two-sample zero-crossing detection in a sinusoidal signal.”
    # Metrologia 49 (2012): 413 - 424. We could also have tried to use more
    # sophisticated interpolation techniques (such as cubic or splines) but it's
    # not clear they would be worth it, given just simple 2x upsampling seems
    # to make the calculations accurate enough for all practical purposes.
    upsampling_ratio = args.upsampling_ratio
    recording_sample_rate *= upsampling_ratio
    test_signal_start_index *= upsampling_ratio
    test_signal_end_index *= upsampling_ratio
    print(
        f"Upsampling recording by {upsampling_ratio}x (to {recording_sample_rate} Hz)",
        file=sys.stderr,
    )
    recording_samples = scipy.signal.resample_poly(
        recording_samples,
        up=upsampling_ratio,
        down=1,
    )
    maybe_write_debug_wavfile("upsampled", recording_samples)

    zero_crossing_indexes = np.nonzero(np.diff(recording_samples > 0))[0]
    recording_slope = np.diff(recording_samples)
    zero_crossing_slopes = recording_slope[zero_crossing_indexes]
    if debug_files_prefix is not None:
        recording_zero_crossing_slopes = np.zeros(recording_samples.size)
        recording_zero_crossing_slopes[zero_crossing_indexes] = zero_crossing_slopes
        maybe_write_debug_wavfile(
            "zero_crossing_slopes", recording_zero_crossing_slopes
        )

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
    #
    # We need to do the above calculations separately for rising and falling
    # edges, because real-world systems often exhibit highly asymmetrical
    # responses where the typical slope (and therefore the threshold we should
    # use) can look quite different between falling and rising edges.
    zero_crossing_partition_nth = int(
        np.round(0.5 * expected_transition_count * args.min_edges_ratio)
    )
    rising_edge_slope_reference = np.partition(
        zero_crossing_slopes[zero_crossing_slopes > 0], -zero_crossing_partition_nth
    )[-zero_crossing_partition_nth]
    falling_edge_slope_reference = np.partition(
        zero_crossing_slopes[zero_crossing_slopes < 0], zero_crossing_partition_nth
    )[zero_crossing_partition_nth]
    assert falling_edge_slope_reference < 0
    rising_edge_slope_threshold = (
        rising_edge_slope_reference * args.edge_slope_threshold
    )
    falling_edge_slope_threshold = (
        falling_edge_slope_reference * args.edge_slope_threshold
    )
    valid_edge_zero_crossing_indexes = np.nonzero(
        (zero_crossing_slopes > rising_edge_slope_threshold)
        | (zero_crossing_slopes < falling_edge_slope_threshold)
    )[0]
    valid_edge_indexes = zero_crossing_indexes[valid_edge_zero_crossing_indexes]
    print(
        f"{zero_crossing_partition_nth}nth steepest slope is {rising_edge_slope_reference} (rising edges) / {falling_edge_slope_reference} (falling edges). Kept {valid_edge_indexes.size} edges (out of {zero_crossing_indexes.size} candidates) whose slope is above {rising_edge_slope_threshold} or below {falling_edge_slope_threshold}. First edge is right after {format_index(valid_edge_indexes[0])} and last edge is right after {format_index(valid_edge_indexes[-1])}.",
        file=sys.stderr,
    )
    assert valid_edge_indexes.size > 1

    # In the typical case, the test video would be crafted in such a way that
    # the instrument sees "grey" (or some equivalent pattern) before and after
    # the test signal to avoid sudden DC shifts in the instrument. (See the
    # padding-related code in the video generator.)
    #
    # If that is the case, then we need to be careful to avoid misinterpreting
    # the initial transition from grey and final transition to grey as real
    # transitions, as these would be spurious transitions as far as the spec is
    # concerned (and their timing would likely be somewhat inconsistent with a
    # true transition between full black and full white).
    #
    # To that end, we look at the first/last N slopes that have the same sign
    # as the very first/last slope, and if that first/last slope is unusually
    # weaker than the others, we get rid of it.
    #
    # Note that we can't just get rid of the first and last transitions
    # unconditionally, for two reasons: (1) it is conceivable that the user may
    # be using a test video without padding; and (2) in some cases the spurious
    # edges are so weak (due to transitioning from/to grey) that they were
    # already rejected in the previous step.
    first_slope_relative = _first_relative_to_same_sign_neighbor_mean(
        zero_crossing_slopes[valid_edge_zero_crossing_indexes],
        args.boundary_edge_rejection_neighbor_count,
    )
    last_slope_relative = _first_relative_to_same_sign_neighbor_mean(
        np.flip(zero_crossing_slopes[valid_edge_zero_crossing_indexes]),
        args.boundary_edge_rejection_neighbor_count,
    )
    print(
        f"First/last edge slopes relative to neighbors min: {first_slope_relative}/{last_slope_relative}",
        file=sys.stderr,
    )
    if first_slope_relative < args.boundary_edge_rejection_slope_threshold:
        valid_edge_zero_crossing_indexes = valid_edge_zero_crossing_indexes[1:]
        valid_edge_indexes = valid_edge_indexes[1:]
    if last_slope_relative < args.boundary_edge_rejection_slope_threshold:
        valid_edge_zero_crossing_indexes = valid_edge_zero_crossing_indexes[:-1]
        valid_edge_indexes = valid_edge_indexes[:-1]

    valid_edge_is_rising = zero_crossing_slopes[valid_edge_zero_crossing_indexes] > 0
    if debug_files_prefix is not None:
        recording_edges = np.zeros(recording_samples.size)
        recording_edges[valid_edge_indexes] = valid_edge_is_rising * 2 - 1
        maybe_write_debug_wavfile("edges", recording_edges)

    # `valid_edge_indexes` refers to the sample right before the zero crossing.
    # This is an integer index whose precision is inherently limited by the
    # sample rate. To improve the precision, we use linear interpolation
    # between that sample and the next to compute a better estimate of the true
    # position of the zero crossing.
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

    edges.to_csv(args.output_frame_transitions_csv_file)


if __name__ == "__main__":
    sys.exit(main())