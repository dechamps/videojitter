#!/usr/bin/env python3

import altair as alt
import argparse
import numpy as np
import json
import pandas as pd
import sys
from si_prefix import si_format
from scipy import stats


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
    )
    argument_parser.add_argument(
        "--chart-minimum-time-between-transitions-seconds",
        help="The minimum time since previous transition that will be shown on the chart before the points are clamped",
        type=float,
        default=0.000,
    )
    argument_parser.add_argument(
        "--chart-maximum-time-between-transitions-seconds",
        help="The maximum time since previous transition that will be shown on the chart before the points are clamped",
        type=float,
        default=0.100,
    )
    argument_parser.add_argument(
        "--black-white-offset-compensation",
        help="Compensate for consistent timing differences between transitions to black vs. transitions to white (usually caused by subtly different black-to-white vs. white-to-black response in the playback system or the recording system). (default: enabled if the spec was generated with a delayed transition)",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    return argument_parser.parse_args()


def interval(series):
    return pd.Interval(series.min(), series.max())


def packed_columns_chart(data, *kargs, **kwargs):
    """Functionally equivalent to `alt.Chart()`, but packs the values in a given
    column together so that the column name only appears once in the Vega-Lite
    spec (instead of once per row). This can massively reduce the size of the
    resulting spec if the number of rows is much larger than the number of
    columns."""
    data = (
        data.apply(lambda column: column.values, result_type="reduce")
        .to_frame()
        .transpose()
    )
    return alt.Chart(data, *kargs, **kwargs).transform_flatten(data.columns.values)


def generate_chart(
    transitions,
    title,
    minimum_time_between_transitions_seconds,
    maximum_time_between_transitions_seconds,
    has_delayed_transitions,
    fine_print,
):
    chart = (
        packed_columns_chart(transitions.reset_index(), title=title)
        .transform_window(transition_count="row_number()")
        .transform_calculate(
            transition_index=alt.expr.datum.transition_count - 1,
            frame_label=alt.expr.if_(alt.datum.frame, "white", "black"),
            label="Transition to " + alt.datum.frame_label,
            opacity=alt.expr.if_(alt.datum.delayed, 0.4, 1),
            delayed_label=alt.expr.if_(
                alt.datum.delayed,
                "Intentionally delayed transition (ignore)",
                "Normal transition",
            ),
            delayed_tooltip_label=alt.expr.if_(alt.datum.delayed, "yes", "no"),
            shape=alt.expr.if_(
                alt.datum.time_since_previous_transition_seconds
                < -minimum_time_between_transitions_seconds,
                "triangle-down",
                alt.expr.if_(
                    alt.datum.time_since_previous_transition_seconds
                    > maximum_time_between_transitions_seconds,
                    "triangle-up",
                    "circle",
                ),
            ),
        )
        .mark_point(filled=True)
        .encode(
            alt.X("recording_timestamp_seconds", type="quantitative")
            .scale(zero=False)
            .axis(
                labelExpr=alt.expr.format(alt.datum.value, "~s") + "s",
                title="Recording timestamp",
            ),
            alt.Y("time_since_previous_transition_seconds", type="quantitative")
            .scale(
                zero=False,
                domain=[
                    -minimum_time_between_transitions_seconds,
                    maximum_time_between_transitions_seconds,
                ],
                clamp=True,
            )
            .axis(
                labelExpr=alt.expr.format(alt.datum.value, "~s") + "s",
                title="Time since previous transition",
            ),
            alt.Color(
                "label",
                type="nominal",
                title=None,
            ).legend(orient="bottom", columns=1, labelLimit=0, clipHeight=15),
            alt.Shape("shape", type="nominal", scale=None),
            tooltip=[
                alt.Tooltip(
                    "transition_index",
                    type="quantitative",
                    title="Recorded transition #",
                ),
                alt.Tooltip(
                    "frame_label",
                    type="nominal",
                    title="Transition to",
                ),
                alt.Tooltip(
                    "recording_timestamp_seconds",
                    title="Recording time (s)",
                    format="~s",
                ),
                alt.Tooltip(
                    "time_since_previous_transition_seconds",
                    title="Time since last transition (s)",
                    format="~s",
                ),
                alt.Tooltip(
                    "delayed_tooltip_label",
                    type="nominal",
                    title="Intentionally delayed",
                ),
            ],
        )
        .properties(width=1000, height=750)
    )
    if has_delayed_transitions:
        chart = chart.encode(
            alt.Opacity("delayed_label", type="nominal", title=None)
            .scale(range=alt.FieldRange("opacity"))
            .legend(orient="bottom", columns=1, labelLimit=0, clipHeight=15),
        )
    return (
        alt.vconcat(
            chart,
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
        .resolve_scale(color="independent", opacity="independent")
        .properties(
            usermeta={
                "embedOptions": {
                    "downloadFileName": "videojitter",
                    # Sets the Vega-Embed PNG export scale factor to provide higher-quality
                    # exports. See https://github.com/vega/vega-embed/issues/492
                    "scaleFactor": 2,
                }
            }
        )
    )


def mean_without_outliers(x):
    return x[np.abs(stats.zscore(x, nan_policy="omit")) < 3].mean()


def estimate_black_lag_seconds(transitions):
    # We don't use the mean to prevent an outlier from biasing all
    # transitions of the same color, nor the median to prevent odd results
    # when dealing with pattern changes.
    return mean_without_outliers(
        transitions.loc[~transitions.frame, "time_since_previous_transition_seconds"]
    ) - mean_without_outliers(
        transitions.loc[transitions.frame, "time_since_previous_transition_seconds"]
    )


def si_format_plus(value, *kargs, **kwargs):
    return ("+" if value >= 0 else "") + si_format(value, *kargs, **kwargs)


def match_delayed_transitions(
    time_since_previous_transition_seconds,
    delayed_transition_indexes,
    expected_transition_count,
):
    transition_count = time_since_previous_transition_seconds.index.size
    if transition_count == expected_transition_count:
        print("Number of recorded transitions matches the spec. Good.", file=sys.stderr)
    else:
        print(
            "WARNING: number of recorded transitions is inconsistent with the spec. Either the recording is corrupted, or the video player skipped/duplicate some transitions entirely. This makes it difficult to figure out which transitions were intentionally delayed, and as such those may be incorrectly annotated.",
            file=sys.stderr,
        )
    delayed_transitions = np.zeros(transition_count, dtype=bool)
    for delayed_transition_index in delayed_transition_indexes:
        # We guess which transition was intentionally delayed using the
        # following heuristic:
        #  1. Based on the spec, define a window over the recorded transitions
        #     where we expect to find the delayed transition;
        #  2. Assume that the delayed transition is the longest one within that
        #     window.
        #
        # If the recorded transition count matches the spec exactly, we assume
        # that the transition sequence matches the spec 1:1 and use a window of
        # size 1 centered on the delayed transition index according to the spec.
        #
        # If the recorded transition count is different from the spec, we use
        # a window whose size is the delta between the transition counts. If
        # there are too many recorded transitions, the window starts on the
        # delayed transition index according to the spec, and we look forward
        # for the delayed transition. If there are too few recorded transitions,
        # the window ends on the delayed transition index according to the spec,
        # and we look backward for the delayed transition.
        window_begin = delayed_transition_index
        window_end = max(
            delayed_transition_index + transition_count - expected_transition_count, 0
        )
        if window_end < window_begin:
            window_begin, window_end = window_end, window_begin
        delayed_transitions[
            time_since_previous_transition_seconds.iloc[
                window_begin : window_end + 1
            ].argmax()
            + window_begin
        ] = True
    return delayed_transitions


def generate_report():
    args = parse_arguments()

    output_chart_file = getattr(args, "output_chart_file", None)
    output_csv = args.output_csv
    assert (
        output_chart_file or output_csv
    ), "At least one of --output-chart-file or --output-csv must be specified"

    spec = json.load(args.spec_file)
    nominal_fps = spec["fps"]["num"] / spec["fps"]["den"]
    transition_count = spec["transition_count"]
    print(
        f"Successfully loaded spec file containing {transition_count} frame transitions at {nominal_fps} FPS",
        file=sys.stderr,
    )

    transitions = pd.read_csv(
        sys.stdin,
        index_col="recording_timestamp_seconds",
        usecols=["recording_timestamp_seconds", "frame"],
    )
    transitions_interval_seconds = interval(transitions.index)
    print(
        f"Recording analysis contains {transitions.index.size} frame transitions, with first transition at {transitions_interval_seconds.left} seconds and last transition at {transitions_interval_seconds.right} seconds for a total of {transitions_interval_seconds.length} seconds",
        file=sys.stderr,
    )

    transitions[
        "time_since_previous_transition_seconds"
    ] = transitions.index.to_series().diff()

    delayed_transitions = spec["delayed_transitions"]
    transitions["delayed"] = match_delayed_transitions(
        transitions["time_since_previous_transition_seconds"],
        delayed_transitions,
        transition_count,
    )

    if getattr(args, "black_white_offset_compensation", delayed_transitions):
        black_lag_seconds = estimate_black_lag_seconds(
            transitions[~transitions.delayed]
        )
        black_offset_seconds = -black_lag_seconds / 2
        white_offset_seconds = black_lag_seconds / 2
        black_white_offset_fineprint = f"Time since last transition includes {si_format_plus(black_offset_seconds, 3)}s correction in all transitions to white and {si_format_plus(white_offset_seconds, 3)}s correction in all transitions to black"
        transitions.loc[
            ~transitions.frame, "time_since_previous_transition_seconds"
        ] += black_offset_seconds
        transitions.loc[
            transitions.frame, "time_since_previous_transition_seconds"
        ] += white_offset_seconds
    else:
        black_white_offset_fineprint = "Consistent timing differences between black vs. white transitions have NOT been compensated for"

    non_delayed_transitions = transitions[~transitions.delayed]
    time_between_transitions_standard_deviation_seconds = (
        non_delayed_transitions.time_since_previous_transition_seconds.std()
    )
    print(
        f"Non-delayed transition interval standard deviation: {time_between_transitions_standard_deviation_seconds} seconds",
        file=sys.stderr,
    )

    if output_csv:
        transitions.to_csv(sys.stdout)
    if output_chart_file:
        minimum_time_between_transitions_index = (
            non_delayed_transitions.time_since_previous_transition_seconds.idxmin()
        )
        maximum_time_between_transitions_index = (
            non_delayed_transitions.time_since_previous_transition_seconds.idxmax()
        )
        mean_time_between_transitions = (
            non_delayed_transitions.time_since_previous_transition_seconds.mean()
        )
        mean_fps = 1 / mean_time_between_transitions
        generate_chart(
            transitions,
            f"{transitions.index.size} transitions at {nominal_fps:.3f} nominal FPS",
            args.chart_minimum_time_between_transitions_seconds,
            args.chart_maximum_time_between_transitions_seconds,
            has_delayed_transitions=delayed_transitions,
            fine_print=[
                f"First transition recorded at {si_format(transitions_interval_seconds.left, 3)}s; last: {si_format(transitions_interval_seconds.right, 3)}s; length: {si_format(transitions_interval_seconds.length, 3)}s",
                f"Recorded {transitions.index.size} transitions; expected {spec['transition_count']} transitions",
                f"{len(delayed_transitions)} transitions were intentionally delayed, and are not included in the below stats:",
                black_white_offset_fineprint,
                f"Transition interval range: {si_format(non_delayed_transitions.loc[minimum_time_between_transitions_index, 'time_since_previous_transition_seconds'], 3)}s (at {si_format(minimum_time_between_transitions_index, 3)}s) to {si_format(non_delayed_transitions.loc[maximum_time_between_transitions_index, 'time_since_previous_transition_seconds'], 3)}s (at {si_format(maximum_time_between_transitions_index, 3)}s) - standard deviation: {si_format(time_between_transitions_standard_deviation_seconds, 3)}s - 99% of transitions are between {si_format(non_delayed_transitions.time_since_previous_transition_seconds.quantile(0.005), 3)}s and {si_format(non_delayed_transitions.time_since_previous_transition_seconds.quantile(0.995), 3)}s",
                f"Mean time between transitions: {si_format(mean_time_between_transitions, 3)}s, i.e. {mean_fps:.06f} FPS, which is {mean_fps/nominal_fps:.6f}x faster than expected (clock skew)",
                f"{(np.abs(stats.zscore(non_delayed_transitions.loc[:, 'time_since_previous_transition_seconds'], nan_policy='omit')) > 3).sum()} transitions are outliers (more than 3 standard deviations away from the mean)",
                "Generated by videojitter",
            ],
        ).save(output_chart_file)


generate_report()
