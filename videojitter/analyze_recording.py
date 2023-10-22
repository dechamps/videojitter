import argparse
import numpy as np
import json
import pandas as pd
import soundfile
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
        "--recording-file",
        help="Path to the input recording file",
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
        "--min-edge-separation-seconds",
        help="The approximate minimum time interval between edges that the analyzer is expected to resolve. If the interval between two edges is shorter than this, the analyzer might become unable to precisely estimate timing differences between the two edges; if the interval is half as short, both edges may be missed entirely. On the other hand, setting this lower lets more high frequency noise in, possibly causing edge detection false positives.",
        type=float,
        default=0.001,
    )
    argument_parser.add_argument(
        "--min-frequency-ratio",
        help='The cutoff frequency of the highpass filter, relative to the nominal FPS. This determines how "sluggish" the system response (including the instrument) is allowed to be. Setting this lower will allow slower transitions to be detected, but lets more low frequency noise in, possibly causing other transitions to be missed or mistimed due to drift.',
        type=float,
        # The default preserves the expected fundamental and gets rid of
        # anything below that.
        default=0.5 / np.sqrt(2),
    )
    argument_parser.add_argument(
        "--boundaries-signal-seconds",
        help="The length of the reference signal used to detect the beginning and end of the test signal within the recording.",
        type=float,
        default=0.5,
    )
    argument_parser.add_argument(
        "--boundaries-score-threshold-ratio",
        help="How well does a given portion of the recording have to match the reference sequence in order for it to be considered as the beginning or end of the test signal, as a ratio of the best match anywhere in the recording.",
        type=float,
        default=0.4,
    )
    argument_parser.add_argument(
        "--upsampling-ratio",
        help="How much to upsample the signal before attempting to find edges. Upsampling reduces interpolation error, thereby improving the precision of zero crossing calculations (timestamp and slope), but makes processing slower.",
        type=int,
        default=2,
    )
    argument_parser.add_argument(
        "--min-edges-ratio",
        help="The minimum number of edges that can be assumed to be present in the test signal, as a ratio of the number of transitions implied by the spec. Used in combination with --edge-amplitude-threshold.",
        type=float,
        default=0.6,
    )
    argument_parser.add_argument(
        "--slope-prominence-threshold",
        help="The absolute slope peak prominence threshold above which an edge will be recorded, as a ratio of the Nth highest prominence, where N is dictated by --min-edges-ratio. Determines how sensitive the analyzer is when detecting edges.",
        type=float,
        default=0.8,
    )
    argument_parser.add_argument(
        "--output-debug-files-prefix",
        help="If set, will write a bunch of files that describe the internal state of the analyzer at various stages of the pipeline under the specified file name prefix. Interpreting this data requires some familiarity with analyzer internals.",
        default=argparse.SUPPRESS,
    )
    return argument_parser.parse_args()


def _generate_boundaries_reference_samples(
    length_seconds, fps_num, fps_den, sample_rate
):
    return videojitter.util.generate_fake_samples(
        np.tile([False, True], int(np.ceil(0.5 * length_seconds * fps_num / fps_den))),
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


def _find_abs_peaks_with_prominence(x):
    """Like scipy.signal.find_peaks(), but returns both positive and negative
    peaks, along with their prominences.

    Negative prominences will be returned for negative peaks.

    Note that the absolute prominences returned by this function are not the
    same as those that would be returned by scipy.signal.find_peaks(np.abs(x)).
    Indeed, when calculating peaks on a fully rectified signal, prominence is
    not increased by having to go through negative values before reaching the
    next peak, but with this function it is."""
    positive_peak_indexes, positive_peak_properties = scipy.signal.find_peaks(
        x, prominence=(None, None)
    )
    negative_peak_indexes, negative_peak_properties = scipy.signal.find_peaks(
        -x, prominence=(None, None)
    )
    return np.concatenate(
        (positive_peak_indexes, negative_peak_indexes)
    ), np.concatenate(
        (
            positive_peak_properties["prominences"],
            -negative_peak_properties["prominences"],
        )
    )


def _interpolate_peaks(x, peak_indexes):
    """Given `peak_indexes` the indexes of samples closest to peaks in `x`,
    returns sub-sample peak location estimates.
    """
    # Simplest way to do this is to use quadratic interpolation. See
    # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Peak_Interpolation.html
    alpha = x[peak_indexes - 1]
    beta = x[peak_indexes]
    gamma = x[peak_indexes + 1]
    return peak_indexes + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)


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

    recording_samples, recording_sample_rate = soundfile.read(
        args.recording_file, dtype=np.float32
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

    min_frequency = nominal_fps * args.min_frequency_ratio
    print(
        f"Removing frequencies lower than {min_frequency} Hz from the recording",
        file=sys.stderr,
    )

    highpass_kernel = _generate_highpass_kernel(min_frequency, recording_sample_rate)
    maybe_write_debug_wavfile("highpass_kernel", highpass_kernel)
    recording_samples = scipy.signal.convolve(
        recording_samples, highpass_kernel.astype(recording_samples.dtype), "same"
    )
    maybe_write_debug_wavfile("highpassed", recording_samples)

    boundaries_reference_samples = _generate_boundaries_reference_samples(
        args.boundaries_signal_seconds,
        spec["fps"]["num"],
        spec["fps"]["den"],
        recording_sample_rate,
    )
    maybe_write_debug_wavfile("boundaries_reference", boundaries_reference_samples)

    cross_correlation = scipy.signal.correlate(
        recording_samples,
        boundaries_reference_samples.astype(recording_samples.dtype)
        / boundaries_reference_samples.size,
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

    # The fundamental, core idea behind the way we detect "edges" is to look for
    # large scale changes in overall recording signal level.
    #
    # One approach is to look at zero crossings; indeed, we can reasonably
    # assume that, on a properly highpassed signal, every true edge will result
    # in a change of sign. However, sadly the converse isn't true: due to high
    # frequency noise the signal will tend to "hover" around zero between edges,
    # creating spurious zero crossings. Separating the spurious zero crossings
    # from the true ones is challenging; using a thresold on the slope at the
    # zero crossing helps, but still fails in pathological cases such as noise
    # causing the signal to "hesitate" (i.e. form a narrow plateau) around the
    # zero crossing, which can result in the edge being missed. (We could work
    # around that by computing the slope using more neighboring points, but that
    # would require us to assume that an edge is more than 2 points wide - which
    # basically amounts to lowpassing the signal. This would impair our ability
    # to resolve fast "blinks".)
    #
    # A more promising idea is to compute the "slope" of the signal and find the
    # steepest slopes. This approach is more flexible than looking only at zero
    # crossings because it will find the highest rate of change even if it is
    # located outside of a zero crossing. (In a sinusoid the highest rate of
    # change happens at the zero crossing, but the recordings we are dealing
    # with are not sinusoids. Even after highpass filtering, the steepest slope
    # might not be located at the zero crossing. This is especially true given
    # real recordings tend to have highly asymmetric shapes: in this case it's
    # mathematically impossible for the steepest slope to be located at the
    # zero crossing on *both* rising and falling edges at the same time.)
    #
    # In practice, what we do is we apply a peak finding algorithm to locate the
    # points at which the signal reaches a local steepness extremum.
    #
    # One thing to watch out for is how that "slope" is computed. The simplest
    # way to compute the slope would be to differentiate the signal (as in,
    # np.diff()). Unfortunately, differentiation also acts as a highpass filter,
    # causing high frequency noise to be amplified tremendously. (Intuitively,
    # this is because the rate of change of a signal is proportional to
    # frequency, not just amplitude.) This is a big problem; for example, high
    # frequency PWM would normally be distinguishable from true edges by its
    # lower amplitude, but since its frequency is much higher, a highpass filter
    # can greatly impact our ability to discriminate between the two.
    #
    # To solve this problem, we can think of differentiation as a convolution
    # with a [-1, +1] kernel. From a Fourier perspective, this operation looks
    # like a highpass filter combined with a 90° phase shift. So, if we want
    # differentation without the highpass filter, what we're really asking for
    # is just the 90° phase shift. Mathematically we land directly on the
    # definition of the Hilbert transform, so that's what we use here.
    recording_slope = -np.imag(scipy.signal.hilbert(recording_samples))
    maybe_write_debug_wavfile("slope", recording_slope)

    # High frequency noise can cause us to find spurious local extrema as the
    # signal "wiggles around" the true peak - this results in a "forest" of
    # peaks (with similar heights) appearing around the true edge. If we don't
    # do anything about this, we will incorrectly report many closely spaced
    # edges for each true edge.
    #
    # In each of these "forests", we need a way to select the tallest peak and
    # ignore the others. Prominence is the ideal metric for this: basically, it
    # indicates how far the signal has to "swing back" before a higher peak is
    # reached in either direction. For the tallest peak in a forest, the only
    # way to get to a higher peak is to go through the previous or next edge.
    # Since that's by definition an opposite edge, we're looking at a full
    # falling+rising edge swing, and the resulting prominence is the entire
    # peak-to-peak amplitude between these edges - hence, very high. In
    # contrast, if the peak is not the tallest in the forest, then by definition
    # there is a higher peak in the same forest, and getting there merely
    # requires a small "hop" through the noise. The prominence is therefore
    # merely the peak-to-peak amplitude of the noise, which in reasonable
    # recordings is expected to be much lower.
    slope_peak_indexes, slope_prominences = _find_abs_peaks_with_prominence(
        recording_slope
    )
    if debug_files_prefix is not None:
        recording_slope_heights = np.zeros(recording_samples.size)
        recording_slope_heights[slope_peak_indexes] = recording_slope[
            slope_peak_indexes
        ]
        maybe_write_debug_wavfile("slope_heights", recording_slope_heights)
        recording_slope_prominence = np.zeros(recording_samples.size)
        recording_slope_prominence[slope_peak_indexes] = slope_prominences
        maybe_write_debug_wavfile("slope_prominences", recording_slope_prominence)

    # We need to decide on a prominence threshold for what constitutes a "true"
    # edge. The correct threshold must be below the minimum true edge peak
    # prominence, which which don't know, so we'll have to take a guess. We
    # could reference our guess on the maximum prominence among all peaks -
    # that's pretty much guaranteed to be a valid edge - but that would make the
    # threshold very sensitive to an isolated outlier. To avoid this problem we
    # base our guess on the Nth peak prominence, where N is large enough to
    # mitigate the influence of outliers, but small enough that we can be
    # reasonably confident we're still going to pick a valid edge even if the
    # test signal contains fewer edges than expected. We then multiply that
    # reference with a fudge factor to allow for edges with slightly smaller
    # prominences, and that's our threshold.
    #
    # We need to do the above calculations separately for rising and falling
    # edges, because real-world systems often exhibit highly asymmetrical
    # responses where the prominence can look quite different between falling
    # and rising edges. TODO: revisit this assumption - it doesn't seem to make
    # sense now that we're using the prominence as a metric, because the
    # prominence is the peak-to-peak amplitude between adjacent edges, and
    # peak-to-peak amplitude by definition doesn't depend on the direction of
    # the edge is calculated from.
    positive_slope_prominences = slope_prominences[slope_prominences > 0]
    negative_slope_prominences = slope_prominences[slope_prominences < 0]
    minimum_onesided_edge_count = 0.5 * expected_transition_count * args.min_edges_ratio
    assert positive_slope_prominences.size >= minimum_onesided_edge_count
    assert negative_slope_prominences.size >= minimum_onesided_edge_count
    positive_slope_prominence_threshold = (
        np.quantile(
            positive_slope_prominences,
            1 - minimum_onesided_edge_count / positive_slope_prominences.size,
        )
        * args.slope_prominence_threshold
    )
    negative_slope_prominence_threshold = (
        np.quantile(
            negative_slope_prominences,
            minimum_onesided_edge_count / negative_slope_prominences.size,
        )
        * args.slope_prominence_threshold
    )
    valid_slope_peak = (slope_prominences > positive_slope_prominence_threshold) | (
        slope_prominences < negative_slope_prominence_threshold
    )
    slope_peak_indexes = slope_peak_indexes[valid_slope_peak]
    slope_prominences = slope_prominences[valid_slope_peak]
    print(
        f"Kept {slope_peak_indexes.size} slope peaks whose prominence is above ~{positive_slope_prominence_threshold:.3} or below ~{negative_slope_prominence_threshold:.3}. First edge is right after {format_index(slope_peak_indexes[0])} and last edge is right after {format_index(slope_peak_indexes[-1])}.",
        file=sys.stderr,
    )
    assert slope_peak_indexes.size > 1

    edge_is_rising = slope_prominences > 0
    if debug_files_prefix is not None:
        recording_edges = np.zeros(recording_samples.size)
        recording_edges[slope_peak_indexes] = edge_is_rising * 2 - 1
        maybe_write_debug_wavfile("edges", recording_edges)

    edges = pd.Series(
        edge_is_rising,
        index=pd.Index(
            (
                _interpolate_peaks(recording_slope, slope_peak_indexes)
                + test_signal_start_index
            )
            / recording_sample_rate,
            name="recording_timestamp_seconds",
        ),
        name="frame",
    )
    edges.sort_index(inplace=True)

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
