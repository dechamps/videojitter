import numpy as np
from videojitter import _signal


def generate_windows(indexes, lookback, lookahead):
    """For each integer in `indexes`, returns a 2D array consisting of a list of windows
    that each contain the `lookback` preceding indexes and `lookahead` following
    indexes.

    For example, generate_windows([10, 20], -2, 1) returns:
    [
        [ -8,  -9, 10, 11],
        [-18, -19, 20, 21],
    ]"""
    return indexes[:, None] + np.arange(-lookback, lookahead + 1)[None, :]


def generate_frames(transition_count, delayed_transitions):
    """Generates an alternating frame sequence in the form of a boolean array.

    In the returned array, False represents black frames and True represents white
    frames. The resulting array always starts with a black frame; it will also end on a
    black frame iff `transition_count` is even.

    The returned array alternates between black and white frames `transition_count`
    times. A black frame is always immediately followed by a white frame and vice-versa,
    except for the transitions whose indexes are specified in `delayed_transitions`,
    which will come after a single repeated frame.

    For example, `generate_frames(6, [4])` will return FTFTFFTF, i.e. alternating
    between black and white 6 times, with the 4th transition (counting from zero) coming
    after a repeated black frame."""
    frames = np.ones(transition_count + len(delayed_transitions) + 1, dtype=int)
    frames[
        np.array(delayed_transitions, dtype=int)
        + 1  # because transition #0 occurs on frame #1
        + np.arange(len(delayed_transitions))
    ] = 0
    return ~(frames.cumsum() % 2).astype(bool)


def generate_fake_recording(frames, fps_num, fps_den, sample_rate, frame_offsets=0):
    """Generates a recording signal simulating what an ideal instrument would output
    when faced with the given frame sequence.
    """
    return _signal.Signal(
        samples=np.repeat(
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
        - 1,
        sample_rate=sample_rate,
    )
