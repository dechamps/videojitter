import numpy as np

import sys


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


def generate_fake_samples(
    frames, fps_num, fps_den, sample_rate, white_duration_overshoot=0
):
    """Generates a recording simulating what an ideal instrument would output
    when faced with the given frame sequence.
    """
    return (
        np.repeat(
            frames,
            np.diff(
                np.round(
                    (np.arange(frames.size) + frames * white_duration_overshoot)
                    * (sample_rate * fps_den / fps_num)
                ).astype(int),
                prepend=0,
            ),
        )
        * 2
        - 1
    )
