import numpy as np


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

    For example, `generate_frames(6, [4])` will return FTFTTFTF, i.e.
    alternating between black and white 6 times, with the 4th transition
    (counting from zero) coming after a repeated white frame."""
    repeated_frames = np.array(delayed_transitions, dtype=int)
    repeated_frames += np.arange(repeated_frames.size)
    frames = np.ones(transition_count + repeated_frames.size + 1, dtype=int)
    frames[repeated_frames] = 0
    frames = (frames.cumsum() - 1) % 2
    return frames.astype(bool)
