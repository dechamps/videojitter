#!/usr/bin/env python3

import altair as alt
import argparse
import numpy as np
import json
import pandas as pd
import scipy.signal
import sys


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given a spec file, and recording analysis results passed in stdin, produces a summary of the results."
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        type=argparse.FileType(),
    )
    argument_parser.add_argument(
        "--output-file",
        help="Path to the output file where the summary will be written. Format is determined from the file extension. Recommended format is HTML. Information on other formats is available at https://altair-viz.github.io/user_guide/saving_charts.html",
        required=True,
    )
    return argument_parser.parse_args()


def interval(series):
    return pd.Interval(series.min(), series.max())


def generate_report():
    args = parse_arguments()

    spec = json.load(args.spec_file)
    nominal_fps = spec["fps"]["num"] / spec["fps"]["den"]
    frame_duration = spec["fps"]["den"] / spec["fps"]["num"]
    reference_frames = 1 * np.array(spec["frames"])
    reference_transitions = pd.Series(
        np.diff(reference_frames),
        index=np.arange(1, reference_frames.size) * frame_duration,
    )
    reference_transitions = reference_transitions[reference_transitions != 0]
    reference_transitions_interval_seconds = interval(reference_transitions.index)
    print(
        f"Successfully loaded spec file containing {reference_transitions.size} frame transitions at {nominal_fps} FPS, with first transition at {reference_transitions_interval_seconds.left} seconds and last transition at {reference_transitions_interval_seconds.right} seconds for a total of {reference_transitions_interval_seconds.length} seconds",
        file=sys.stderr,
    )

    transitions = pd.read_csv(
        sys.stdin,
        index_col="recording_timestamp_seconds",
        usecols=["recording_timestamp_seconds", "frame"],
    ).squeeze()
    transitions_interval_seconds = interval(transitions.index)
    print(
        f"Recording analysis contains {transitions.size} frame transitions, with first transition at {transitions_interval_seconds.left} seconds and last transition at {transitions_interval_seconds.right} seconds for a total of {transitions_interval_seconds.length} seconds",
        file=sys.stderr,
    )
    if transitions.size == reference_transitions.size:
        print("Number of recorded transitions matches the spec. Good.", file=sys.stderr)
    else:
        print(
            "WARNING: number of recorded transitions is inconsistent with the spec. Expect garbage results.",
            file=sys.stderr,
        )

    # TODO: try harder to make this work when there are more or fewer
    # transitions than expected. reindex(method="nearest") might help.
    transitions = transitions.to_frame()
    transitions.loc[:, "error_seconds"] = (
        transitions.index - reference_transitions.index
    )
    linear_regression = np.polynomial.Polynomial.fit(
        transitions.index, transitions.loc[:, "error_seconds"], deg=1
    )
    clock_skew = 1 + linear_regression.coef[1]
    if abs(linear_regression.coef[1]) > 0.10:
        print(
            f"WARNING: abnormally large clock skew detected - recording is {clock_skew}x longer than expected.",
            file=sys.stderr,
        )
    else:
        print(
            f"Recording is {clock_skew}x longer than expected. This is usually due to benign clock skew. Scaling timestamps to compensate.",
            file=sys.stderr,
        )
    transitions.loc[:, "error_seconds"] -= linear_regression(transitions.index)

    transitions.loc[:, "frame_and_last_duration"] = (
        transitions.loc[:, "frame"]
        + " after "
        + np.insert(
            np.round(np.diff(reference_transitions.index), 4) * 1000,
            0,
            [np.nan],
        ).astype(str)
        + " ms"
    )

    alt.Chart(transitions.reset_index()).mark_point().encode(
        alt.X("recording_timestamp_seconds").scale(zero=False),
        alt.Y("error_seconds").scale(zero=False),
        alt.Color("frame_and_last_duration"),
    ).properties(width=1000, height=750).save(args.output_file)


generate_report()
