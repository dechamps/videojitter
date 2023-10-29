import numpy as np
import scipy.signal


def generate_windows(indexes, lookback, lookahead):
    """For each integer in `indexes`, returns a 2D array consisting of
    a list of windows that each contain the `lookback` preceding indexes and
    `lookahead` following indexes.

    For example, generate_windows([10, 20], -2, 1) returns:
    [
        [ -8,  -9, 10, 11],
        [-18, -19, 20, 21],
    ]"""
    return indexes[:, None] + np.arange(-lookback, lookahead + 1)[None, :]


def generate_frames(transition_count, delayed_transitions):
    """Generates an alternating frame sequence in the form of a boolean array.

    In the returned array, False represents black frames and True represents
    white frames. The resulting array always starts with a black frame; it will
    also end on a black frame iff `transition_count` is even.

    The returned array alternates between black and white frames
    `transition_count` times. A black frame is always immediately followed by
    a white frame and vice-versa, except for the transitions whose indexes are
    specified in `delayed_transitions`, which will come after a single repeated
    frame.

    For example, `generate_frames(6, [4])` will return FTFTFFTF, i.e.
    alternating between black and white 6 times, with the 4th transition
    (counting from zero) coming after a repeated black frame."""
    frames = np.ones(transition_count + len(delayed_transitions) + 1, dtype=int)
    frames[
        np.array(delayed_transitions, dtype=int)
        + 1  # because transition #0 occurs on frame #1
        + np.arange(len(delayed_transitions))
    ] = 0
    return ~(frames.cumsum() % 2).astype(bool)


def generate_fake_samples(frames, fps_num, fps_den, sample_rate, frame_offsets=0):
    """Generates a recording simulating what an ideal instrument would output
    when faced with the given frame sequence.
    """
    return (
        np.repeat(
            frames,
            np.diff(
                np.round(
                    (np.arange(frames.size) + (1 + frame_offsets))
                    * (sample_rate * fps_den / fps_num)
                ).astype(np.int64),
                prepend=0,
            ),
        ).astype(np.int8)
        * 2
        - 1
    )


def firwin(*kargs, pass_zero=True, **kwargs):
    """Equivalent to scipy.signal.firwin() but with a workaround for
    the `pass_zero=False` bug described at
    https://github.com/scipy/scipy/issues/19291.
    """
    kernel = scipy.signal.firwin(*kargs, **kwargs)
    if not pass_zero:
        kernel = -kernel
        kernel[int(kernel.size / 2)] += 1
    return kernel
