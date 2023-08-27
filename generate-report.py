#!/usr/bin/env python3

import altair as alt
import argparse
import numpy as np
import json
import pandas as pd
import scipy.signal
import sys
from si_prefix import si_format


def parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given a spec file, and recording analysis results passed in stdin, produces a summary of the results.",
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
        "--output-chart-file",
        help="Path to the output file where the chart will be written. Format is determined from the file extension. Recommended format is HTML. Information on other formats is available at https://altair-viz.github.io/user_guide/saving_charts.html",
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--output-csv",
        help="Output the results as CSV to standard output",
        action="store_true",
        default=False,
    )
    argument_parser.add_argument(
        "--compensate-clock-skew",
        help="Calculate and compensate for clock skew, i.e. the difference in speed between the recording clock and the video clock that would otherwise result in a sloped/tilted graph. Note that the clock skew estimate can be incorrect (e.g. for recordings where the overall mean error undergoes sudden changes), leading to odd results.",
        action="store_true",
        default=False,
    )
    argument_parser.add_argument(
        "--display-maximum-absolute-error-seconds",
        help="The maximum absolute error that will be shown on the graph before the points are clamped",
        type=float,
        default=0.050,
    )
    return argument_parser.parse_args()


def interval(series):
    return pd.Interval(series.min(), series.max())


def rescale(series, target_range):
    original_range = interval(series)
    return (
        series - original_range.left
    ) / original_range.length * target_range.length + target_range.left


def match_transitions(transitions, reference_transitions):
    """Join actual transitions in `transitions` against expected transitions in
    `reference_transitions`.

    The join is done based on frame (black or white) and proximity of normalized
    timestamps. Columns in `reference_transitions` will be prefixed with
    `reference_`.

    If a given reference transition is the best match for multiple actual
    transitions, the reference transition information is duplicated into all
    matching rows and the `duplicate` column is set to True for all these rows.

    If a given reference transition is not the best match for any actual
    transition, the reference transition information is inserted into a new row
    where all actual transition information is set to NaN/NA.

    The return value also includes a `expected_recording_timestamp_seconds`
    column which indicates where we would have expected to find the transition
    in a "perfect" recording. This is useful to locate missing transitions.
    """
    transitions = transitions.reset_index()
    transitions.index = pd.Index(
        rescale(
            transitions.loc[:, "recording_timestamp_seconds"],
            interval(reference_transitions.index),
        ),
        name="scaled_recording_timestamp_seconds",
    )

    def filter_frame(transitions_to_filter, frame):
        return transitions_to_filter.loc[
            transitions_to_filter.loc[:, "frame"] == frame, :
        ]

    def match_transitions_for_frame(frame):
        reference_transitions_for_merge = filter_frame(reference_transitions, frame)
        return pd.merge(
            left=reference_transitions_for_merge.rename(
                lambda column_name: "reference_" + column_name, axis="columns"
            ),
            left_index=True,
            right=pd.merge_asof(
                left=filter_frame(transitions, frame),
                left_index=True,
                right=reference_transitions_for_merge.index.to_series().rename(
                    "reference_timestamp_seconds"
                ),
                right_index=True,
                direction="nearest",
            ),
            right_on="reference_timestamp_seconds",
            how="outer",
        )

    transitions = pd.concat(
        [match_transitions_for_frame(frame) for frame in [False, True]]
    )
    transitions.sort_index(inplace=True)
    transitions.loc[:, "duplicate"] = transitions.loc[
        :, "reference_timestamp_seconds"
    ].duplicated(keep=False)
    transitions.loc[:, "expected_recording_timestamp_seconds"] = rescale(
        transitions.loc[:, "reference_timestamp_seconds"],
        interval(transitions.loc[:, "recording_timestamp_seconds"]),
    )
    return transitions


def error_linear_regression(transitions, deg):
    # TODO: find a better way to calculate clock skew so that we can enable
    # clock skew compensation by default. The current method produces
    # nonsensical results in some cases; for example the slope part of the
    # linear regression breaks down if the mean suddenly changes in the middle
    # of the recording.
    valid_transitions = transitions.loc[
        ~(
            pd.isna(transitions.loc[:, "recording_timestamp_seconds"])
            | transitions.loc[:, "duplicate"]
        ),
        :,
    ]
    return np.polynomial.Polynomial.fit(
        valid_transitions.loc[:, "recording_timestamp_seconds"],
        valid_transitions.loc[:, "error_seconds"],
        deg=deg,
    )


def generate_chart(
    transitions, nominal_fps, maximum_absolute_error_seconds, fine_print
):
    chart = alt.Chart(transitions)
    return alt.vconcat(
        (
            chart.properties(
                title=f"{int(transitions.loc[:, 'transition_index'].max())+1} transitions at {nominal_fps:.3f} nominal FPS"
            )
            .transform_calculate(
                anomaly=alt.expr.if_(
                    alt.datum["duplicate"],
                    "Duplicate transition",
                    alt.expr.if_(
                        alt.expr.isValid(alt.datum["recording_timestamp_seconds"]),
                        None,
                        "Missing transition",
                    ),
                ),
            )
            .transform_filter(alt.expr.isValid(alt.datum["anomaly"]))
            .mark_rule(strokeWidth=2)
            .encode(
                alt.X("estimated_recording_timestamp_seconds", type="quantitative"),
                alt.Color("anomaly", type="nominal", title=None)
                .scale(
                    domain=["Missing transition", "Duplicate transition"],
                    range=["orangered", "orange"],
                )
                .legend(orient="bottom", columns=2, labelLimit=0, symbolStrokeWidth=3),
            )
            + chart.transform_calculate(
                reference_frame_label=alt.expr.if_(
                    alt.datum["reference_frame"], "white", "black"
                ),
                duplicate_label=alt.expr.if_(alt.datum["duplicate"], "yes", "no"),
                label="Transition to "
                + alt.datum["reference_frame_label"]
                + " (after "
                + alt.datum["reference_previous_frame_count"]
                + " "
                + alt.expr.if_(alt.datum["reference_frame"], "black", "white")
                + " frames)",
                shape=alt.expr.if_(
                    alt.datum["error_seconds"] < -maximum_absolute_error_seconds,
                    "triangle-down",
                    alt.expr.if_(
                        alt.datum["error_seconds"] > maximum_absolute_error_seconds,
                        "triangle-up",
                        "circle",
                    ),
                ),
            )
            .mark_point(filled=True)
            .encode(
                alt.X("estimated_recording_timestamp_seconds", type="quantitative")
                .scale(zero=False)
                .axis(
                    labelExpr=alt.expr.format(alt.datum["value"], "~s") + "s",
                    title="Recording timestamp",
                ),
                alt.Y("error_seconds")
                .scale(
                    zero=False,
                    domain=[
                        -maximum_absolute_error_seconds,
                        maximum_absolute_error_seconds,
                    ],
                    clamp=True,
                )
                .axis(
                    labelExpr=alt.expr.format(alt.datum["value"], "+~s") + "s",
                    title="Transition timing error",
                ),
                alt.Color(
                    "label",
                    type="nominal",
                    title=None,
                ).legend(orient="bottom", columns=2, labelLimit=0),
                alt.Shape("shape", type="nominal", scale=None),
                tooltip=[
                    alt.Tooltip("transition_index", title="Recorded transition #"),
                    alt.Tooltip(
                        "reference_transition_index", title="Reference transition #"
                    ),
                    alt.Tooltip("reference_frame_index", title="Reference frame #"),
                    alt.Tooltip(
                        "reference_frame_label",
                        type="nominal",
                        title="Transition to",
                    ),
                    alt.Tooltip(
                        "reference_previous_frame_count",
                        type="nominal",
                        title="Frames since last transition",
                    ),
                    alt.Tooltip(
                        "duplicate_label",
                        type="nominal",
                        title="Duplicate transition",
                    ),
                    alt.Tooltip(
                        "reference_timestamp_seconds",
                        title="Reference time (seconds)",
                        format="~s",
                    ),
                    alt.Tooltip(
                        "recording_timestamp_seconds",
                        title="Recording time (seconds)",
                        format="~s",
                    ),
                    alt.Tooltip(
                        "error_seconds",
                        title="Timing error (seconds)",
                        format="+~s",
                    ),
                ],
            )
        )
        .transform_calculate(
            estimated_recording_timestamp_seconds=alt.expr.if_(
                alt.expr.isValid(alt.datum["recording_timestamp_seconds"]),
                alt.datum["recording_timestamp_seconds"],
                alt.datum["expected_recording_timestamp_seconds"],
            )
        )
        .resolve_scale(color="independent")
        .properties(width=1000, height=750),
        alt.Chart(
            title=alt.TitleParams(
                fine_print,
                fontSize=10,
                fontWeight="lighter",
                color="gray",
                anchor="start",
            )
        ).mark_text(),
    )


def generate_report():
    args = parse_arguments()

    output_chart_file = getattr(args, "output_chart_file", None)
    output_csv = args.output_csv
    assert (
        output_chart_file or output_csv
    ), "At least one of --output-chart-file or --output-csv must be specified"

    spec = json.load(args.spec_file)
    nominal_fps = spec["fps"]["num"] / spec["fps"]["den"]
    frame_duration = spec["fps"]["den"] / spec["fps"]["num"]
    reference_transitions = pd.DataFrame(
        {"frame": spec["frames"], "frame_index": np.arange(0, len(spec["frames"]))},
        index=pd.Index(
            np.arange(0, len(spec["frames"])) * frame_duration,
            name="timestamp_seconds",
        ),
    )
    reference_transitions_diff = (
        reference_transitions.loc[:, "frame"]
        != reference_transitions.loc[:, "frame"].shift()
    )
    reference_transitions_diff[0] = False
    reference_transitions.loc[:, "previous_frame_count"] = (
        reference_transitions.groupby(reference_transitions_diff.cumsum())
        .cumcount()
        .shift()
        + 1
    )
    reference_transitions = reference_transitions[reference_transitions_diff]
    reference_transitions.loc[:, "transition_index"] = np.arange(
        0, reference_transitions.index.size
    )
    reference_transitions_interval_seconds = interval(reference_transitions.index)
    print(
        f"Successfully loaded spec file containing {reference_transitions.size} frame transitions at {nominal_fps} FPS, with first transition at {reference_transitions_interval_seconds.left} seconds and last transition at {reference_transitions_interval_seconds.right} seconds for a total of {reference_transitions_interval_seconds.length} seconds",
        file=sys.stderr,
    )

    transitions = pd.read_csv(
        sys.stdin,
        index_col="recording_timestamp_seconds",
        usecols=["recording_timestamp_seconds", "frame"],
    )
    transitions.loc[:, "transition_index"] = np.arange(0, transitions.size)
    transitions_interval_seconds = interval(transitions.index)
    print(
        f"Recording analysis contains {transitions.index.size} frame transitions, with first transition at {transitions_interval_seconds.left} seconds and last transition at {transitions_interval_seconds.right} seconds for a total of {transitions_interval_seconds.length} seconds",
        file=sys.stderr,
    )
    if transitions.index.size == reference_transitions.index.size:
        print("Number of recorded transitions matches the spec. Good.", file=sys.stderr)
    else:
        print(
            "WARNING: number of recorded transitions is inconsistent with the spec. Either the recording is corrupted, or the video player skipped/duplicate some transitions entirely.",
            file=sys.stderr,
        )

    transitions = match_transitions(transitions, reference_transitions)

    transitions.loc[:, "error_seconds"] = (
        transitions.loc[:, "recording_timestamp_seconds"]
        - transitions.loc[:, "reference_timestamp_seconds"]
    )
    linear_regression = error_linear_regression(
        transitions, deg=1 if args.compensate_clock_skew else 0
    )
    if args.compensate_clock_skew:
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
    transitions.loc[:, "error_seconds"] -= linear_regression(
        transitions.loc[:, "recording_timestamp_seconds"]
    )

    frames = transitions.loc[:, "frame"].astype(bool)
    black_offset = transitions.loc[~frames, "error_seconds"].mean()
    white_offset = transitions.loc[frames, "error_seconds"].mean()
    print(
        f"Offsets black: {white_offset} seconds white: {black_offset} seconds",
        file=sys.stderr,
    )
    transitions.loc[~frames, "error_seconds"] -= black_offset
    transitions.loc[frames, "error_seconds"] -= white_offset

    error_standard_deviation = transitions.loc[:, "error_seconds"].std()
    print(
        f"Error standard deviation: {error_standard_deviation} seconds",
        file=sys.stderr,
    )

    if output_csv:
        transitions.to_csv(sys.stdout)
    if output_chart_file:
        error_minimum_index = transitions.loc[:, "error_seconds"].idxmin()
        error_maximum_index = transitions.loc[:, "error_seconds"].idxmax()
        generate_chart(
            transitions,
            nominal_fps,
            args.display_maximum_absolute_error_seconds,
            fine_print=[
                f"First transition recorded at {si_format(transitions_interval_seconds.left, 3)}s; last: {si_format(transitions_interval_seconds.right, 3)}s; length: {si_format(transitions_interval_seconds.length, 3)}s",
                f"First transition reference timestamp is {si_format(reference_transitions_interval_seconds.left, 3)}s; last: {si_format(reference_transitions_interval_seconds.right, 3)}s; length: {si_format(reference_transitions_interval_seconds.length, 3)}s",
                f"Recorded {int(transitions.loc[:, 'transition_index'].max())+1} transitions; expected {int(transitions.loc[:, 'reference_transition_index'].max())+1} reference transitions across {int(transitions.loc[:, 'reference_frame_index'].max())+1} frames",
                f"Detected {transitions.loc[:, 'duplicate'].sum()} duplicate transitions and {pd.isna(transitions.loc[:, 'recording_timestamp_seconds']).sum()} missing transitions",
                f"Compensating for estimated recording clock skew of ~{clock_skew:.6f}x"
                if args.compensate_clock_skew
                else "Recording clock skew compensation is disabled",
                f"Timing error range: {si_format(transitions.loc[error_minimum_index, 'error_seconds'], 3)}s (at {si_format(transitions.loc[error_minimum_index, 'recording_timestamp_seconds'], 3)}s) to {si_format(transitions.loc[error_maximum_index, 'error_seconds'], 3)}s (at {si_format(transitions.loc[error_maximum_index, 'recording_timestamp_seconds'], 3)}s) - standard deviation: {si_format(error_standard_deviation, 3)}s - 99% of transitions are between {si_format(transitions.loc[:, 'error_seconds'].quantile(0.005), 3)}s and {si_format(transitions.loc[:, 'error_seconds'].quantile(0.995), 3)}s",
                "Generated by videojitter",
            ],
        ).save(output_chart_file)


generate_report()
