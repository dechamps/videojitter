import argparse
import sys
import json
import numpy as np
import soundfile
import scipy.special
import videojitter.util


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Generates a light waveform recording faking what a real instrument would output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--output-recording-file",
        help="Write the fake recording to the specified WAV file",
        required=True,
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
        "--padding-signal-level",
        help="The signal level of the padding (before --dc-offset, --invert and --amplitude). -1, 0, and 1 can be used to simulate black, grey and white padding, respectively.",
        type=float,
        default=0.2,
    )
    argument_parser.add_argument(
        "--clock-skew",
        help="Simulate clock skew, i.e. the test signal will be stretched by this amount. Note this doesn't affect padding.",
        type=float,
        default=0.95,
    )
    argument_parser.add_argument(
        "--pattern-count",
        help="Modulates the frame durations in such a way as to create a visible pattern on the resulting charts. This occurs before the overshoots are added. This option sets the number of times the pattern repeats; fractional numbers can be used to add padding at the beginning and end of the test signal. Set to zero to disable.",
        type=float,
        default=3.5,
    )
    argument_parser.add_argument(
        "--pattern-min-interval",
        help="The minimum frame interval to use when generating the pattern (see --pattern-count), as a ratio of the nominal frame duration.",
        type=float,
        default=0.5,
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
        "--pwm-frequency-fps",
        help="Modulate the light waveform with PWM at this frequency times nominal FPS. Used to simulate PWM brightness modulation from real displays. Set to zero to disable.",
        type=float,
        default=7.6,  # Non-round to ensure PWM does not coincide with frames
    )
    argument_parser.add_argument(
        "--pwm-duty-cycle",
        help="PWM duty cycle. See --pwm-frequency-fps.",
        type=float,
        default=0.9,
    )
    argument_parser.add_argument(
        "--dc-offset",
        help="Add a DC offset.",
        type=float,
        default=1.0,
    )
    argument_parser.add_argument(
        "--invert",
        help="Invert the test signal (after the DC offset), i.e. white is low and black is high",
        action="store_true",
    )
    argument_parser.add_argument(
        "--amplitude",
        help="Amplitude of the resulting signal, where 1.0 (plus the DC offset) is full scale.",
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
    argument_parser.add_argument(
        "--noise-rms-per-hz",
        help="Add gaussian noise of the specified RMS amplitude multiplied by half the sample rate. This is done as the last step. Set to zero to disable.",
        type=float,
        default=0.00000005,
    )
    argument_parser.add_argument(
        "--output-sample-type",
        help='Output sample format as a python-soundfile subtype, e.g. "PCM_16", "PCM_24", "FLOAT".',
        default="FLOAT",
    )
    return argument_parser.parse_args()


def _apply_gaussian_filter(samples, stddev_samples):
    kernel = scipy.signal.windows.gaussian(
        M=int(np.round(stddev_samples * 10)),
        std=stddev_samples,
    ).astype(samples.dtype)
    return scipy.signal.convolve(
        samples,
        kernel / np.sum(kernel),
        mode="same",
    )


def _get_pattern_frame_offset_adjustments(frame_count, pattern_count, start):
    # The goal here is to generate a pattern that looks like a "sawtooth" on the
    # resulting chart. From a mathematical perspective this is surprisingly
    # tricky, because when we change the duration of a frame, this doesn't just
    # change the position of the following transition on the Y axis - it also
    # changes its position on the X axis, since the timestamp of the transition
    # changes. The change in the X axis is much less visible than the change in
    # the Y axis due to the difference in scale, but it still results in a
    # curved line, especially for tall patterns (small `start`). We want a
    # perfectly straight line to make it easier to tell when the analyzer is
    # malfunctioning.
    #
    # More rigorously, the value on the Y axis is equal to the difference with
    # the previous point on the X axis:
    #
    #   y(n)=dx(n)=x(n)-x(n-1)
    #
    # In order to draw in a straight line, we want the change in Y to equal the
    # change in X (times a constant A):
    #
    #   y(n)=A*dx(n)
    #
    # Combining the above:
    #
    #   y(n)=(y(n)-y(n-1))/A
    #
    # Which reduces to:
    #
    #   y(n)=B*C^(n-1)
    #
    # That's not all. The above is a way to compute the interval between two
    # successive frames, but what we really need is a way to compute the
    # adjustments to individual frame timestamps (offsets). Indeed the
    # adjustment to a given frame timestamp has to take into account the
    # adjustments to all previous frame timestamps. This is given by the
    # integral of y(n), which we'll note Y(n).
    #
    # But wait, we're still not done. Where this becomes really tricky is that
    # we want the pattern to have a neutral effect on the overall duration of
    # the frame sequence, i.e. when all the adjustments to frame durations are
    # summed up, the result should be zero. Otherwise this would mess up the
    # average FPS and might also result in discontinuities at the beginning
    # and/or end of the pattern. This means we have to constrain the parameters
    # such that Y(n) is zero at the end of the period. Setting a constraint on
    # the period itself would not work well because the parameter has limited
    # resolution (it's an integer). Instead we let the user choose the period as
    # well as the initial/minimum frame duration adjustment (`start`) and from
    # there we calculate the maximum frame duration adjustment (`end`).
    #
    # Coming up with an equation for `end` such that the final value of Y(n) is
    # zero is surprisingly hard. See the generator-pattern.ipynb Jupyter
    # notebook for an overview of the math that was used to arrive at the
    # formulas for Y(n) (`frame_offset_adjustments`) and `end` that are used in
    # this code.
    period_frames = int(np.floor(frame_count / pattern_count))
    offsetIntoCycle = np.arange(0, period_frames) / period_frames
    end = -np.real(scipy.special.lambertw(-start * np.exp(-start), -1))
    frame_offset_adjustments = np.tile(
        (
            start * ((end / start) ** offsetIntoCycle - 1) / np.log(end / start)
            - offsetIntoCycle
        )
        * period_frames,
        int(pattern_count),
    )

    frame_index = int((frame_count - frame_offset_adjustments.size) / 2)
    all_frame_offset_adjustments = np.zeros(frame_count)
    all_frame_offset_adjustments[
        frame_index : frame_index + frame_offset_adjustments.size
    ] = frame_offset_adjustments
    return all_frame_offset_adjustments


def main():
    args = _parse_arguments()
    sample_rate = args.internal_sample_rate_hz
    with open(args.spec_file) as spec_file:
        spec = json.load(spec_file)

    assert args.internal_sample_rate_hz > args.output_sample_rate_hz
    downsample_ratio = int(
        np.ceil(args.internal_sample_rate_hz / args.output_sample_rate_hz)
    )
    sample_rate = args.output_sample_rate_hz * downsample_ratio
    print(f"Using internal sample rate of {sample_rate} Hz", file=sys.stderr)

    frames = videojitter.util.generate_frames(
        spec["transition_count"], spec["delayed_transitions"]
    )
    samples = videojitter.util.generate_fake_samples(
        frames,
        spec["fps"]["num"],
        spec["fps"]["den"],
        sample_rate / args.clock_skew,
        frame_offsets=(
            _get_pattern_frame_offset_adjustments(
                frames.size,
                args.pattern_count,
                args.pattern_min_interval,
            )
            if args.pattern_count
            else 0
        )
        + frames * args.white_duration_overshoot
        + (np.arange(frames.size) % 2 == 0) * args.even_duration_overshoot,
    )
    samples = np.concatenate(
        (
            np.ones(
                int(np.round(args.begin_padding_seconds * sample_rate)),
                dtype=samples.dtype,
            )
            * args.padding_signal_level,
            samples,
            np.ones(
                int(np.round(args.end_padding_seconds * sample_rate)),
                dtype=samples.dtype,
            )
            * args.padding_signal_level,
        )
    ).astype(np.float32)
    if args.pwm_frequency_fps != 0:
        samples = (samples + 1) * (
            scipy.signal.square(
                np.arange(samples.size)
                * (
                    2
                    * np.pi
                    * args.pwm_frequency_fps
                    * spec["fps"]["num"]
                    / spec["fps"]["den"]
                    / sample_rate
                ),
                args.pwm_duty_cycle,
            )
            * 0.5
            + 0.5
        ) - 1
    samples = (
        scipy.signal.resample_poly(
            samples,
            up=1,
            down=downsample_ratio,
        )
        + args.dc_offset
    ) * ((-1 if args.invert else 1) * args.amplitude)
    sample_rate = sample_rate / downsample_ratio

    if args.gaussian_filter_stddev_seconds:
        gaussian_filter_stddev_samples = (
            args.gaussian_filter_stddev_seconds * sample_rate
        )
        samples = _apply_gaussian_filter(samples, gaussian_filter_stddev_samples)

    if args.high_pass_filter_hz:
        samples = scipy.signal.sosfilt(
            scipy.signal.butter(
                1, args.high_pass_filter_hz, "highpass", fs=sample_rate, output="sos"
            ),
            samples,
        ).astype(samples.dtype)

    if args.noise_rms_per_hz:
        samples += np.random.default_rng(0).normal(
            scale=args.noise_rms_per_hz * sample_rate / 2, size=samples.size
        )

    soundfile.write(
        args.output_recording_file,
        samples,
        samplerate=int(sample_rate),
        subtype=args.output_sample_type,
    )


if __name__ == "__main__":
    sys.exit(main())
