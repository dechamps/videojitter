#!/usr/bin/env python3

import altair as alt
import argparse
import numpy as np
import json
import pandas as pd
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
    return argument_parser.parse_args()


def interval(series):
    return pd.Interval(series.min(), series.max())


def generate_chart(
    transitions,
    title,
    minimum_time_between_transitions_seconds,
    maximum_time_between_transitions_seconds,
    fine_print,
):
    return (
        alt.vconcat(
            alt.Chart(transitions.reset_index(), title=title)
            .transform_window(transition_count="row_number()")
            .transform_calculate(
                transition_index=alt.expr.datum["transition_count"] - 1,
                frame_label=alt.expr.if_(alt.datum["frame"], "white", "black"),
                label="Transition to " + alt.datum["frame_label"],
                shape=alt.expr.if_(
                    alt.datum["error_seconds"]
                    < -minimum_time_between_transitions_seconds,
                    "triangle-down",
                    alt.expr.if_(
                        alt.datum["error_seconds"]
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
                    labelExpr=alt.expr.format(alt.datum["value"], "~s") + "s",
                    title="Recording timestamp",
                ),
                alt.Y("time_since_previous_transition_seconds")
                .scale(
                    zero=False,
                    domain=[
                        -minimum_time_between_transitions_seconds,
                        maximum_time_between_transitions_seconds,
                    ],
                    clamp=True,
                )
                .axis(
                    labelExpr=alt.expr.format(alt.datum["value"], "+~s") + "s",
                    title="Time since previous transition",
                ),
                alt.Color(
                    "label",
                    type="nominal",
                    title=None,
                ).legend(orient="bottom", columns=2, labelLimit=0),
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
                ],
            )
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
        .resolve_scale(color="independent")
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
    if transitions.index.size == transition_count:
        print("Number of recorded transitions matches the spec. Good.", file=sys.stderr)
    else:
        print(
            "WARNING: number of recorded transitions is inconsistent with the spec. Either the recording is corrupted, or the video player skipped/duplicate some transitions entirely.",
            file=sys.stderr,
        )

    transitions.loc[
        :, "time_since_previous_transition_seconds"
    ] = transitions.index.to_series().diff()

    time_between_transitions_standard_deviation_seconds = transitions.loc[
        :, "time_since_previous_transition_seconds"
    ].std()
    print(
        f"Transition interval standard deviation: {time_between_transitions_standard_deviation_seconds} seconds",
        file=sys.stderr,
    )

    if output_csv:
        transitions.to_csv(sys.stdout)
    if output_chart_file:
        minimum_time_between_transitions_index = transitions.loc[
            :, "time_since_previous_transition_seconds"
        ].idxmin()
        maximum_time_between_transitions_index = transitions.loc[
            :, "time_since_previous_transition_seconds"
        ].idxmax()
        generate_chart(
            transitions,
            f"{transitions.index.size} transitions at {nominal_fps:.3f} nominal FPS",
            args.chart_minimum_time_between_transitions_seconds,
            args.chart_maximum_time_between_transitions_seconds,
            fine_print=[
                f"First transition recorded at {si_format(transitions_interval_seconds.left, 3)}s; last: {si_format(transitions_interval_seconds.right, 3)}s; length: {si_format(transitions_interval_seconds.length, 3)}s",
                f"Recorded {transitions.index.size} transitions; expected {spec['transition_count']} transitions",
                f"Transition interval range: {si_format(transitions.loc[minimum_time_between_transitions_index, 'time_since_previous_transition_seconds'], 3)}s (at {si_format(minimum_time_between_transitions_index, 3)}s) to {si_format(transitions.loc[maximum_time_between_transitions_index, 'time_since_previous_transition_seconds'], 3)}s (at {si_format(maximum_time_between_transitions_index, 3)}s) - standard deviation: {si_format(time_between_transitions_standard_deviation_seconds, 3)}s - 99% of transitions are between {si_format(transitions.loc[:, 'time_since_previous_transition_seconds'].quantile(0.005), 3)}s and {si_format(transitions.loc[:, 'time_since_previous_transition_seconds'].quantile(0.995), 3)}s",
                "Generated by videojitter",
            ],
        ).save(output_chart_file)


generate_report()
