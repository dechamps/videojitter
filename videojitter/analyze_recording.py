import argparse
import json
import sys
import numpy as np
import pandas as pd
import scipy.signal
from videojitter import _signal, _util


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description=(
            "Given the recorded light waveform file, analyzes the recording and writes"
            " the list of detected edges (transitions) to a file."
        ),
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
        "--output-edges-csv-file",
        help="Write the list of detected edges to the specified CSV file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--min-edge-separation-seconds",
        help=(
            "The approximate minimum time interval between edges that the analyzer is"
            " expected to resolve. If the interval between two edges is shorter than"
            " this, the analyzer might become unable to precisely estimate timing"
            " differences between the two edges; if the interval is half as short, both"
            " edges may be missed entirely. On the other hand, setting this lower lets"
            " more high frequency noise in, possibly causing edge detection false"
            " positives."
        ),
        type=float,
        default=0.001,
    )
    argument_parser.add_argument(
        "--min-frequency-ratio",
        help=(
            "The cutoff frequency of the highpass filter, relative to the nominal FPS."
            ' This determines how "sluggish" the system response (including the'
            " instrument) is allowed to be. Setting this lower will allow slower"
            " transitions to be detected, but lets more low frequency noise in,"
            " possibly causing other transitions to be missed or mistimed due to drift."
        ),
        type=float,
        # The default preserves the expected fundamental and gets rid of
        # anything below that.
        default=0.5 / np.sqrt(2),
    )
    argument_parser.add_argument(
        "--pattern-length-seconds",
        help=(
            "The length of the reference pattern used to detect the beginning and end"
            " of the test signal within the recording."
        ),
        type=float,
        default=0.5,
    )
    argument_parser.add_argument(
        "--pattern-score-threshold",
        help=(
            "How well does a given portion of the recording have to match the reference"
            " pattern in order for it to be considered as the beginning or end of the"
            " test signal, as a ratio of the best match anywhere in the recording."
        ),
        type=float,
        default=0.4,
    )
    argument_parser.add_argument(
        "--min-edges-ratio",
        help=(
            "The minimum number of edges that can be assumed to be present in the test"
            " signal, as a ratio of the number of transitions implied by the spec. Used"
            " in combination with --edge-amplitude-threshold."
        ),
        type=float,
        default=0.6,
    )
    argument_parser.add_argument(
        "--slope-prominence-threshold",
        help=(
            "The absolute slope peak prominence threshold above which an edge will be"
            " recorded, as a ratio of the Nth highest prominence, where N is dictated"
            " by --min-edges-ratio. Determines how sensitive the analyzer is when"
            " detecting edges."
        ),
        type=float,
        default=0.6,
    )
    argument_parser.add_argument(
        "--output-debug-files-prefix",
        help=(
            "If set, will write a bunch of files that describe the internal state of"
            " the analyzer at various stages of the pipeline under the specified file"
            " name prefix. Interpreting this data requires some familiarity with"
            " analyzer internals."
        ),
        default=argparse.SUPPRESS,
    )
    return argument_parser.parse_args()


def _generate_pattern_signal(length_seconds, fps_num, fps_den, sample_rate):
    return _util.generate_fake_recording(
        np.tile([False, True], int(np.ceil(0.5 * length_seconds * fps_num / fps_den))),
        fps_num,
        fps_den,
        sample_rate,
    )


def _generate_slope_kernel(min_frequency_hz, sample_rate):
    # The simplest way to compute signal slope would be to differentiate the signal (as
    # in, np.diff()). Unfortunately, differentiation also acts as a highpass filter with
    # a constant +6dB/octave slope, causing high frequency noise to be amplified
    # tremendously. (Intuitively, this is because the rate of change of a signal is
    # proportional to frequency, not just amplitude.) This is a big problem; for
    # example, high frequency PWM would normally be distinguishable from true edges by
    # its lower amplitude, but since its frequency is much higher, a highpass filter can
    # greatly impact our ability to discriminate between the two.
    #
    # To solve this problem, we can think of differentiation as a convolution with a
    # [-1, +1] kernel. From a Fourier perspective, this operation looks like a highpass
    # filter combined with a 90° phase shift. So, if we want differentation without the
    # highpass filter, what we're really asking for is just the 90° phase shift.
    #
    # A 90° phase shift happens to be the mathematical definition of the Hilbert
    # transform, so that's what we end up with here. This code is conceptually
    # equivalent to -np.imag(scipy.signal.hilbert(x)), which provides a "perfect"
    # frequency response, but is inefficient as it computes an FFT over the entire
    # input. We can get practically equivalent results by convolving with a relatively
    # short kernel. This generates a type III FIR hilbert transformer following the
    # principles described at:
    #   https://en.wikipedia.org/wiki/Hilbert_transform#Discrete_Hilbert_transform
    half_length = int(np.ceil(sample_rate / min_frequency_hz / 2)) * 2
    slope_kernel = np.zeros(half_length * 2 + 1)
    slope_kernel[1::2] = (-2 / np.pi) / np.arange(
        start=-half_length + 1, stop=half_length + 1, step=2
    )
    # Apply a gentle highpass filter to reject frequencies we don't care about. We use a
    # very short filter to mitigate ringing artefacts, which could be mistaken for
    # edges.
    kernel = _signal.convolve(
        _signal.Signal(samples=slope_kernel, sample_rate=sample_rate),
        _signal.firwin(
            numtaps=half_length * 2 + 1,
            cutoff=min_frequency_hz,
            pass_zero=False,
            window="boxcar",
            sample_rate=sample_rate,
        ),
        "same",
    )
    return kernel._replace(
        samples=kernel.samples * scipy.signal.windows.blackmanharris(slope_kernel.size)
    )


def _find_peaks_with_prominence_mirrored(x):
    """Like scipy.signal.find_peaks(), but calculates prominences under the assumption
    that the borders are mirror images of the signal; this ensures that the first/last
    peaks do not end up with an unfairly low prominence just because they don't have an
    opposite peak on *both* sides."""
    # This makes sure that, if scipy.signal.peak_prominences() hits a signal border, the
    # resulting base will always be lower than the one on the other side of the peak.
    # Therefore, the lowest contour line (which determines the prominence) will always
    # be on the other side of the peak.
    #
    # Note that using a lower value (such as -infinity) would result in incorrect
    # prominence for the tallest peak in the signal, which is computed based on the
    # global minimum.
    xmin = np.min(x)
    x = np.insert(x, [0, x.size], [xmin, xmin])

    peak_indexes, peak_properties = scipy.signal.find_peaks(x, prominence=(None, None))

    # Our modification may have created spurious peaks at the 2nd and/or second-to-last
    # samples, so make sure to reject those.
    valid_peak = (peak_indexes > 1) & (peak_indexes < x.size - 2)

    return peak_indexes[valid_peak] - 1, peak_properties["prominences"][valid_peak]


def _find_abs_peaks_with_prominence(x):
    """Like _find_peaks_with_prominence_mirrored(), but considers both positive and
    negative peaks.

    Negative prominences will be returned for negative peaks.

    Note that the absolute prominences returned by this function are not the same as
    those that would be returned by _find_peaks_with_prominence_mirrored(np.abs(x)).
    Indeed, when calculating peaks on a fully rectified signal, prominence is not
    increased by having to go through negative values before reaching the next peak, but
    with this function it is."""
    (
        positive_peak_indexes,
        positive_peak_prominences,
    ) = _find_peaks_with_prominence_mirrored(x)
    (
        negative_peak_indexes,
        negative_peak_prominences,
    ) = _find_peaks_with_prominence_mirrored(-x)
    return np.concatenate(
        (
            positive_peak_indexes,
            negative_peak_indexes,
        )
    ), np.concatenate(
        (
            positive_peak_prominences,
            -negative_peak_prominences,
        )
    )


def _interpolate_peaks(x, peak_indexes):
    """Given `peak_indexes` the indexes of samples closest to peaks in `x`, returns
    sub-sample peak location estimates.
    """
    # Simplest way to do this is to use quadratic interpolation. See
    # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Peak_Interpolation.html
    alpha = x[peak_indexes - 1]
    beta = x[peak_indexes]
    gamma = x[peak_indexes + 1]
    return peak_indexes + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)


def _format_index(signal, index):
    return f"sample {index} ({index / signal.sample_rate} seconds)"


def _generate_sparse_signal_from_template(template, indexes, values):
    samples = np.zeros(template.samples.size)
    samples[indexes] = values
    return _signal.Signal(samples=samples, sample_rate=template.sample_rate)


class _Analyzer:
    def __init__(self, args):
        self._args = args
        with open(args.spec_file, encoding="utf-8") as spec_file:
            self._spec = json.load(spec_file)
        self._debug_wavfile_index = 0

    def analyze(self):
        expected_duration_seconds = (
            self._spec["transition_count"] + len(self._spec["delayed_transitions"])
        ) / self._nominal_fps()
        print(
            "Successfully loaded spec file describing"
            f" {self._spec['transition_count']} transitions (of which"
            f" {len(self._spec['delayed_transitions'])} are delayed) at"
            f" {self._nominal_fps()} FPS ({expected_duration_seconds} seconds)",
            file=sys.stderr,
        )

        recording = _signal.fromfile(self._args.recording_file, dtype=np.float32)
        print(
            f"Successfully loaded recording containing {recording.samples.size} samples"
            f" at {recording.sample_rate} Hz ({_signal.duration(recording)} seconds)",
            file=sys.stderr,
        )
        self._detect_clipping(recording)
        recording = self._downsample(recording)

        test_signal_start_index, test_signal_end_index = self._find_boundaries(
            recording
        )
        recording = recording._replace(
            samples=recording.samples[test_signal_start_index:test_signal_end_index]
        )
        self._write_debug_wavfile("trimmed", lambda: recording)

        edge_positions, edge_is_rising = self._detect_edges(recording)
        self._write_csv(
            (edge_positions + test_signal_start_index) / recording.sample_rate,
            edge_is_rising,
        )

    def _detect_clipping(self, recording):
        max_index = np.argmax(np.abs(recording.samples))
        if recording.samples[max_index] < 0.9:
            return
        print(
            "WARNING: it looks like the recording may be clipping around"
            f" {_format_index(recording, max_index)}. You may want to re-record at a"
            " lower input gain/volume.",
            file=sys.stderr,
        )

    def _downsample(self, recording):
        # If we assume the worst-case scenario where all the edges in the input signal
        # are separated by the min threshold Tm, then the fundamental period in the
        # input signal is 2*Tm (because the period is a full rising edge + falling edge
        # cycle), and the fundamental frequency is 1/(2*Tm). However we also need to
        # preserve the second harmonic, because that carries the information about
        # waveform asymmetry - if we lose that, then it will look like every rising edge
        # occurs exactly halfway between every falling edge (and vice-versa), thus
        # destroying important timing information that the user may care about (consider
        # e.g. 3:2 patterns). Therefore we need to preserve frequencies up to 1/Tm, and
        # thus, per Nyquist, we need a sampling rate of at least 2/Tm.
        downsampling_ratio = np.floor(
            0.5 * self._args.min_edge_separation_seconds * recording.sample_rate
        )
        print(
            f"Downsampling recording by {downsampling_ratio}x (to"
            f" {recording.sample_rate / downsampling_ratio} Hz)",
            file=sys.stderr,
        )
        recording = _signal.downsample(recording, downsampling_ratio)
        self._write_debug_wavfile("downsampled", lambda: recording)
        return recording

    def _find_boundaries(self, recording):
        pattern = self._generate_pattern(recording.sample_rate)
        boundary_candidate_indexes = self._extract_boundary_candidate_indexes(
            self._correlate_pattern(recording, pattern)
        )
        assert boundary_candidate_indexes.size > 1
        test_signal_start_index = boundary_candidate_indexes[0]
        test_signal_end_index = boundary_candidate_indexes[-1] + pattern.samples.size
        print(
            "Test signal appears to start at"
            f" {_format_index(recording, test_signal_start_index)} and end at"
            f" {_format_index(recording, test_signal_end_index)} in the"
            " recording.",
            file=sys.stderr,
        )
        if (
            test_signal_start_index < recording.sample_rate
            and test_signal_end_index > recording.samples.size - recording.sample_rate
        ):
            print(
                "WARNING: test signal boundaries are very close to recording"
                " boundaries. This may mean the recording is truncated or the"
                " boundaries were not detected correctly (e.g. because the recording is"
                " corrupted or doesn't match the spec). This warning can be ignored the"
                " recording was trimmed manually (you shouldn't need to do that though"
                " - the analyzer can detect where the test signal begins and ends"
                " automatically!).",
                file=sys.stderr,
            )
        return test_signal_start_index, test_signal_end_index

    def _generate_pattern(self, sample_rate):
        pattern = _generate_pattern_signal(
            self._args.pattern_length_seconds,
            self._spec["fps"]["num"],
            self._spec["fps"]["den"],
            sample_rate,
        )
        self._write_debug_wavfile("pattern", lambda: pattern)
        return pattern

    def _correlate_pattern(self, recording, pattern):
        correlation = _signal.correlate(
            recording,
            pattern._replace(
                samples=(pattern.samples / pattern.samples.size).astype(
                    recording.samples.dtype
                )
            ),
            mode="valid",
        )
        self._write_debug_wavfile(
            "pattern_correlation",
            lambda: correlation,
        )
        return correlation._replace(samples=np.abs(correlation.samples))

    def _extract_boundary_candidate_indexes(self, pattern_correlation):
        boundary_candidates = pattern_correlation._replace(
            samples=pattern_correlation.samples
            >= np.max(pattern_correlation.samples) * self._args.pattern_score_threshold
        )
        self._write_debug_wavfile("boundary_candidates", lambda: boundary_candidates)
        return np.nonzero(boundary_candidates.samples)[0]

    def _detect_edges(self, recording):
        slope = self._generate_recording_slope(recording)
        edge_indexes, edge_is_rising = self._detect_edges_from_slope_peaks(slope)
        return (
            _interpolate_peaks(slope.samples, edge_indexes),
            edge_is_rising,
        )

    def _generate_recording_slope(self, recording):
        min_frequency = self._nominal_fps() * self._args.min_frequency_ratio
        print(
            f"Removing frequencies lower than {min_frequency} Hz from the recording",
            file=sys.stderr,
        )

        # The fundamental, core idea behind the way we detect "edges" is to look for
        # large scale changes in overall recording signal level.
        #
        # One approach is to look at zero crossings; indeed, we can reasonably assume
        # that, on a properly highpassed signal, every true edge will result in a change
        # of sign. However, sadly the converse isn't true: due to high frequency noise
        # the signal will tend to "hover" around zero between edges, creating spurious
        # zero crossings. Separating the spurious zero crossings from the true ones is
        # challenging; using a thresold on the slope at the zero crossing helps, but
        # still fails in pathological cases such as noise causing the signal to
        # "hesitate" (i.e. form a narrow plateau) around the zero crossing, which can
        # result in the edge being missed. (We could work around that by computing the
        # slope using more neighboring points, but that would require us to assume that
        # an edge is more than 2 points wide - which basically amounts to lowpassing the
        # signal. This would impair our ability to resolve fast "blinks".)
        #
        # A more promising idea is to compute the "slope" of the signal and find the
        # steepest slopes. This approach is more flexible than looking only at zero
        # crossings because it will find the highest rate of change even if it is
        # located outside of a zero crossing. (In a sinusoid the highest rate of change
        # happens at the zero crossing, but the recordings we are dealing with are not
        # sinusoids. Even after highpass filtering, the steepest slope might not be
        # located at the zero crossing. This is especially true given real recordings
        # tend to have highly asymmetric shapes: in this case it's mathematically
        # impossible for the steepest slope to be located at the zero crossing on *both*
        # rising and falling edges at the same time.)
        #
        # In practice, what we do is we apply a peak finding algorithm to locate the
        # points at which the signal reaches a local steepness extremum.
        kernel = _generate_slope_kernel(min_frequency, recording.sample_rate)
        self._write_debug_wavfile("slope_kernel", lambda: kernel)
        recording_length = recording.samples.size
        recording = recording._replace(
            samples=np.concatenate(
                (
                    np.full(kernel.samples.size // 2, recording.samples[0]),
                    recording.samples,
                    np.full(kernel.samples.size // 2, recording.samples[-1]),
                )
            )
        )
        self._write_debug_wavfile("padded", lambda: recording)
        slope = _signal.convolve(
            recording,
            kernel._replace(samples=kernel.samples.astype(recording.samples.dtype)),
            "valid",
        )
        self._write_debug_wavfile("slope", lambda: slope)
        assert slope.samples.size == recording_length
        return slope

    def _detect_edges_from_slope_peaks(self, slope):
        peak_indexes, peak_prominences = self._find_peaks(slope)

        abs_peak_prominences = np.abs(peak_prominences)
        peak_prominence_threshold = self._get_slope_peak_prominence_threshold(
            abs_peak_prominences
        )
        valid_peak = abs_peak_prominences > peak_prominence_threshold
        peak_indexes = peak_indexes[valid_peak]
        peak_prominences = peak_prominences[valid_peak]
        print(
            f"Kept {peak_indexes.size} slope peaks whose prominence is above"
            f" ~{peak_prominence_threshold:.3}. First edge is right after"
            f" {_format_index(slope, peak_indexes[0])} and last edge is"
            f" right after {_format_index(slope, peak_indexes[-1])}.",
            file=sys.stderr,
        )
        assert peak_indexes.size > 1

        edge_is_rising = peak_prominences > 0
        self._write_debug_wavfile(
            "edges",
            lambda: _generate_sparse_signal_from_template(
                slope, peak_indexes, edge_is_rising * 2 - 1
            ),
        )
        return peak_indexes, edge_is_rising

    def _find_peaks(self, slope):
        # High frequency noise can cause us to find spurious local extrema as the signal
        # "wiggles around" the true peak - this results in a "forest" of peaks (with
        # similar heights) appearing around the true edge. If we don't do anything about
        # this, we will incorrectly report many closely spaced edges for each true edge.
        #
        # In each of these "forests", we need a way to select the tallest peak and
        # ignore the others. Prominence is the ideal metric for this: basically, it
        # indicates how far the signal has to "swing back" before a higher peak is
        # reached in either direction. For the tallest peak in a forest, the only way to
        # get to a higher peak is to go through the previous or next edge. Since that's
        # by definition an opposite edge, we're looking at a full falling+rising edge
        # swing, and the resulting prominence is the entire peak-to-peak amplitude
        # between these edges - hence, very high. In contrast, if the peak is not the
        # tallest in the forest, then by definition there is a higher peak in the same
        # forest, and getting there merely requires a small "hop" through the noise. The
        # prominence is therefore merely the peak-to-peak amplitude of the noise, which
        # in reasonable recordings is expected to be much lower.
        peak_indexes, peak_prominences = _find_abs_peaks_with_prominence(slope.samples)
        self._write_debug_wavfile(
            "slope_heights",
            lambda: _generate_sparse_signal_from_template(
                slope, peak_indexes, slope.samples[peak_indexes]
            ),
        )
        self._write_debug_wavfile(
            "slope_prominences",
            lambda: _generate_sparse_signal_from_template(
                slope, peak_indexes, peak_prominences
            ),
        )
        return peak_indexes, peak_prominences

    def _get_slope_peak_prominence_threshold(self, abs_slope_peak_prominences):
        # We need to decide on a prominence threshold for what constitutes a "true"
        # edge. The correct threshold must be below the minimum true edge peak
        # prominence, which which don't know, so we'll have to take a guess. We could
        # reference our guess on the maximum prominence among all peaks - that's pretty
        # much guaranteed to be a valid edge - but that would make the threshold very
        # sensitive to an isolated outlier. To avoid this problem we base our guess on
        # the Nth peak prominence, where N is large enough to mitigate the influence of
        # outliers, but small enough that we can be reasonably confident we're still
        # going to pick a valid edge even if the test signal contains fewer edges than
        # expected. We then multiply that reference with a fudge factor to allow for
        # edges with slightly smaller prominences, and that's our threshold.
        minimum_edge_count = self._spec["transition_count"] * self._args.min_edges_ratio
        assert abs_slope_peak_prominences.size >= minimum_edge_count
        return (
            np.quantile(
                abs_slope_peak_prominences,
                1 - minimum_edge_count / abs_slope_peak_prominences.size,
            )
            * self._args.slope_prominence_threshold
        )

    def _write_csv(self, edge_timestamps, edge_is_rising):
        edges = pd.Series(
            edge_is_rising,
            index=pd.Index(edge_timestamps, name="recording_timestamp_seconds"),
            name="edge_is_rising",
        )
        edges.sort_index(inplace=True)
        edges.to_csv(self._args.output_edges_csv_file)

    def _write_debug_wavfile(self, name, generate_signal, normalize=False):
        debug_files_prefix = getattr(self._args, "output_debug_files_prefix", None)
        if debug_files_prefix is None:
            return
        signal = generate_signal()
        if normalize:
            signal = signal._replace(
                samples=signal.samples / np.max(np.abs(signal.samples))
            )
        _signal.tofile(
            signal._replace(samples=signal.samples.astype(np.float32)),
            file=f"{debug_files_prefix}{self._debug_wavfile_index:02}_{name}.wav",
        )
        self._debug_wavfile_index += 1

    def _nominal_fps(self):
        return self._spec["fps"]["num"] / self._spec["fps"]["den"]


def main():
    _Analyzer(_parse_arguments()).analyze()


if __name__ == "__main__":
    sys.exit(main())
