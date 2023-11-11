import argparse
import json
import sys
import altair as alt
import numpy as np
import pandas as pd
from si_prefix import si_format
from scipy import stats
import videojitter._util


def _parse_arguments():
    argument_parser = argparse.ArgumentParser(
        description="Given an edges file, produces a summary of the data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argument_parser.add_argument(
        "--spec-file",
        help="Path to the input spec file",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--edges-csv-file",
        help="Path to the CSV file containing the list of edges found in the recording",
        required=True,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--output-chart-file",
        help=(
            "Output the results as a graphical chart to the specified file. Format is"
            " determined from the file extension. Recommended format is HTML."
            " Information on other formats is available at"
            " https://altair-viz.github.io/user_guide/saving_charts.html. Can be"
            " specified multiple times to save to multiple formats at once."
        ),
        default=argparse.SUPPRESS,
        action="append",
    )
    argument_parser.add_argument(
        "--output-csv-file",
        help="Output the results as CSV to standard output",
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--chart-minimum-time-between-transitions-seconds",
        help=(
            "The minimum time since previous transition that will be shown on the chart"
            " before the points are clamped"
        ),
        type=float,
        default=0.000,
    )
    argument_parser.add_argument(
        "--chart-maximum-time-between-transitions-seconds",
        help=(
            "The maximum time since previous transition that will be shown on the chart"
            " before the points are clamped"
        ),
        type=float,
        default=0.100,
    )
    argument_parser.add_argument(
        "--keep-first-transition",
        help=(
            "By default the very first transition is thrown away as it may be a"
            ' spurious transition from the warmup pattern instead of the first "true"'
            " transition. If this option is set, the first transition is preserved."
        ),
        action="store_true",
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--keep-last-transition",
        help=(
            "By default the very last transition is thrown away as it may be a spurious"
            ' transition to the cooldown pattern instead of the last "true" transition.'
            " If this option is set, the last transition is preserved."
        ),
        action="store_true",
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--edge-direction-compensation",
        help=(
            "Compensate for shared timing differences between all falling and rising"
            " edges, i.e. transitions to black vs. transitions to white (usually caused"
            " by subtly different black-to-white vs. white-to-black response in the"
            " playback system or the recording system). (default: enabled if the spec"
            " was generated with a delayed transition)"
        ),
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
    )
    argument_parser.add_argument(
        "--delayed-transition-max-offset",
        help=(
            "How many transitions to use on either side of where we would expect to"
            " find each delayed transition to find the real delayed transition."
        ),
        type=int,
        default=4,
    )
    argument_parser.add_argument(
        "--time-precision-seconds-decimals",
        help=(
            "How many decimals to round to when producing timestamps (and differences"
            " between timestamps). Used to avoid producing overly long floating point"
            " numbers where there is no actual precision benefit."
        ),
        default=6,  # Microsecond precision
    )
    return argument_parser.parse_args()


def _interval(series):
    return pd.Interval(series.min(), series.max())


def _packed_columns_chart(data, *kargs, **kwargs):
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


def _generate_chart(
    transitions,
    title,
    first_transition_recording_timestamp_seconds,
    high_is_white,
    minimum_time_between_transitions_seconds,
    maximum_time_between_transitions_seconds,
    mean_time_between_transitions,
    fine_print,
):
    chart = (
        _packed_columns_chart(
            # Stop Altair from outputting NaNs, which is not valid JSON. See
            # https://github.com/altair-viz/altair/issues/2301
            transitions.replace({np.nan: None}),
            title=title,
            name="chart",
        )
        .transform_window(transition_count="row_number()")
        .transform_calculate(
            time_since_first_transition=alt.datum.recording_timestamp_seconds
            - first_transition_recording_timestamp_seconds,
            transition_index=alt.expr.datum.transition_count - 1,
            edge_label=alt.expr.if_(alt.datum.edge_is_rising, "rising", "falling"),
            **(
                {}
                if high_is_white is None
                else {
                    "frame_label": alt.expr.if_(
                        (
                            alt.datum.edge_is_rising
                            if high_is_white
                            else ~alt.datum.edge_is_rising
                        ),
                        "white",
                        "black",
                    )
                }
            ),
            label=alt.expr.if_(
                alt.datum.valid,
                alt.expr.upper(alt.expr.slice(alt.datum.edge_label, 0, 1))
                + alt.expr.slice(alt.datum.edge_label, 1),
                "Invalid " + alt.datum.edge_label,
            )
            + " edge"
            + (
                ""
                if high_is_white is None
                else alt.expr.if_(
                    alt.datum.valid,
                    " (transition to " + alt.datum.frame_label + ")",
                    "",
                )
            ),
            label_order=(~alt.datum.valid) * 2 + (~alt.datum.edge_is_rising),
            valid_label=alt.expr.if_(alt.datum.valid, "yes", "no"),
            time_since_previous_transition_seconds_relative_to_mean=(
                alt.datum.time_since_previous_transition_seconds
                - mean_time_between_transitions
            ),
        )
        .mark_point(filled=True)
        .encode(
            alt.X("time_since_first_transition", type="quantitative")
            .scale(zero=False, nice=False)
            .axis(
                labelExpr=alt.expr.format(alt.datum.value, "~s") + "s",
                title="Time since first transition",
            ),
            alt.Y("time_since_previous_transition_seconds", type="quantitative")
            .scale(
                zero=False,
                domain=[
                    minimum_time_between_transitions_seconds,
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
                sort=alt.SortField("label_order"),
            )
            .scale(range=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
            .legend(orient="bottom", columns=1, labelLimit=0, clipHeight=15),
            shape={
                # https://github.com/altair-viz/altair/issues/2759
                "condition": [
                    {
                        "test": (
                            alt.datum.time_since_previous_transition_seconds
                            < minimum_time_between_transitions_seconds
                        ),
                        "value": "triangle-down",
                    },
                    {
                        "test": (
                            alt.datum.time_since_previous_transition_seconds
                            > maximum_time_between_transitions_seconds
                        ),
                        "value": "triangle-up",
                    },
                ],
                "value": "circle",
            },
        )
        .properties(width=1000, height=750)
    )
    tooltips = (
        [
            alt.Tooltip(
                "transition_index",
                type="quantitative",
                title="Recorded transition #",
            ),
            alt.Tooltip(
                "edge_label",
                type="nominal",
                title="Edge",
            ),
            alt.Tooltip(
                "valid_label",
                type="nominal",
                title="Valid",
            ),
        ]
        + (
            []
            if high_is_white is None
            else [
                alt.Tooltip(
                    "frame_label",
                    type="nominal",
                    title="Transition to",
                )
            ]
        )
        + [
            alt.Tooltip(
                "recording_timestamp_seconds",
                type="quantitative",
                title="Recording time (s)",
                format="~s",
            ),
            alt.Tooltip(
                "time_since_first_transition",
                type="quantitative",
                title="Time since first transition (s)",
                format="~s",
            ),
            alt.Tooltip(
                "time_since_previous_transition_seconds",
                type="quantitative",
                title="Time since prev. transition (s)",
                format="~s",
            ),
            alt.Tooltip(
                "time_since_previous_transition_seconds_relative_to_mean",
                type="quantitative",
                title="Relative to mean (s)",
                format="+~s",
            ),
        ]
    )
    if "intentionally_delayed" not in transitions:
        chart = chart.encode(opacity=alt.value(1))
    else:
        normal_label = "Normal transition"
        delayed_label = "Intentionally delayed transition (ignore)"
        chart = chart.transform_calculate(
            intentionally_delayed_label=alt.expr.if_(
                alt.datum.intentionally_delayed,
                delayed_label,
                normal_label,
            ),
            intentionally_delayed_tooltip=alt.expr.if_(
                alt.datum.intentionally_delayed, "yes", "no"
            ),
        ).encode(
            alt.Opacity("intentionally_delayed_label", type="nominal", title=None)
            .scale(
                # It would be cleaner to use a field-based range instead of
                # mapping legend labels, but we can't because of this bug:
                # https://github.com/vega/vega-lite/issues/9150
                domain=[normal_label, delayed_label],
                range=[1, 0.5],
            )
            .legend(orient="bottom", columns=1, labelLimit=0, clipHeight=15),
        )
        tooltips.append(
            alt.Tooltip(
                "intentionally_delayed_tooltip",
                type="nominal",
                title="Intentionally delayed",
            )
        )
    return (
        alt.vconcat(
            chart.encode(tooltip=tooltips).add_params(
                # Make the chart zoomable on the X axis.
                # Note we don't let the user zoom the Y axis, as they would then
                # end up scaling both axes simultaneously, which does not really
                # make sense (the aspect ratio of this chart is arbitrary
                # anyway) and is more annoying than useful when attempting to
                # keep outliers within the range of the chart.
                alt.selection_interval(
                    name="x_interval", encodings=["x"], bind="scales"
                ),
            ),
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
                    # Sets the Vega-Embed PNG export scale factor to provide
                    # higher-quality exports. See:
                    #   https://github.com/vega/vega-embed/issues/492
                    "scaleFactor": 2,
                }
            }
        )
    )


def _mean_without_outliers(x):
    return x[np.abs(stats.zscore(x, nan_policy="omit")) < 3].mean()


def _estimate_falling_edge_lag_seconds(transitions):
    # We don't use the mean to prevent an outlier from biasing a given edge
    # direction, nor the median to prevent odd results when dealing with pattern
    # changes.
    return _mean_without_outliers(
        transitions.loc[
            ~transitions.edge_is_rising, "time_since_previous_transition_seconds"
        ]
    ) - _mean_without_outliers(
        transitions.loc[
            transitions.edge_is_rising, "time_since_previous_transition_seconds"
        ]
    )


def _si_format_plus(value, *kargs, **kwargs):
    return ("+" if value >= 0 else "") + si_format(value, *kargs, **kwargs)


def _match_delayed_transitions(
    transitions,
    delayed_transition_indexes,
    expected_transition_count,
    max_offset,
):
    # Locating delayed transitions by index alone will not give satisfactory
    # results if the recording has missing or spurious transitions, as those
    # would offset the index. Instead, calculate the timestamp at which we would
    # expect to find the delayed transition in an ideal recording, then locate
    # the closest real transition. This should work in all but the most
    # pathological cases (e.g. time-varying clock skew).
    delayed_transition_indexes = np.array(delayed_transition_indexes)
    delayed_transition_expected_time_seconds = (
        delayed_transition_indexes + 1 + np.arange(delayed_transition_indexes.size)
    ) / (
        (expected_transition_count + delayed_transition_indexes.size)
        / (
            transitions.recording_timestamp_seconds.max()
            - transitions.recording_timestamp_seconds.min()
        )
    ) + transitions.recording_timestamp_seconds.min()
    recording_delayed_transition_indexes = pd.Index(
        transitions.recording_timestamp_seconds
    ).get_indexer(delayed_transition_expected_time_seconds, method="nearest")

    # In theory we could stop there, but we shouldn't, because the delayed
    # transition can be a few frames off (e.g. if there are spurious extra
    # transitions at the beginning and/or end of the recording). So look at the
    # data to produce a better guess.

    # Look at a few transitions before and after the one that is closest to the
    # expected timestamp.
    neighbors_indexes = videojitter._util.generate_windows(
        recording_delayed_transition_indexes, max_offset, max_offset
    )
    neighbors_durations_seconds = (
        transitions.time_since_previous_transition_seconds.values[neighbors_indexes]
    )

    # First, align even frames with odd frames to prevent bias from affecting
    # the outcome.
    # TODO: it's surprising that this is necessary, and even that it works,
    # given that the point of delayed transitions is precisely to remove this
    # bias??
    even_bias = (
        np.mean(neighbors_durations_seconds[:, ::2], axis=1)
        - np.mean(neighbors_durations_seconds[:, 1::2], axis=1)
    )[:, None]
    neighbors_durations_seconds[:, 0::2] -= even_bias / 2
    neighbors_durations_seconds[:, 1::2] += even_bias / 2

    # Estimate the mean frame duration. The delayed transition counts for 2
    # frames.
    mean_frame_durations_seconds = np.sum(neighbors_durations_seconds, axis=1) / (
        max_offset * 2 + 2
    )

    # A transition is considered a candidate if it happens significantly later
    # than what the mean frame duration would predict.
    candidate_transitions = (
        neighbors_durations_seconds > 1.5 * mean_frame_durations_seconds[:, None]
    )

    # We assume we found the correct delayed transition if it is the only
    # candidate within the window.
    transition_found = np.count_nonzero(candidate_transitions, axis=1) == 1
    not_found_indexes = np.nonzero(~transition_found)[0]
    if not_found_indexes.size > 0:
        print(
            "WARNING: unable to locate the following delayed transitions:"
            f" {delayed_transition_indexes[not_found_indexes]} (expected to find them"
            " around"
            f" {delayed_transition_expected_time_seconds[not_found_indexes]} seconds)."
            " These delayed transitions will not be reported, and black/white color"
            " information may not be available.",
            file=sys.stderr,
        )

    return pd.DataFrame(
        {"expected_transition_index": delayed_transition_indexes[transition_found]},
        pd.Index(
            neighbors_indexes[transition_found][
                candidate_transitions[transition_found]
            ],
            name="transition_index",
        ),
    )


def _is_high_white(transitions):
    """Given zero, one or more transitions with expected transition indexes,
    returns True if high is white (i.e. a rising edge is a transition to a white
    frame, and a falling edge is a transition to a black frame), False is low is
    white (i.e. a rising edge is a transition to a black frame and a falling
    edge is a transition to a white frame), or None if the results are
    inconclusive."""

    # We assume, by convention, that the first transition as defined by the spec
    # is a transition from black to white. That transition has index 0;
    # therefore, an even transition index means a transition from black to
    # white.
    transitions_high_is_white = transitions.edge_is_rising == (
        transitions.expected_transition_index % 2 == 0
    )

    # Only return a conclusive result if all transitions agree.
    high_is_white = transitions_high_is_white.values.all()
    low_is_white = (~transitions_high_is_white.values).all()
    return None if high_is_white == low_is_white else high_is_white


def main():
    args = _parse_arguments()

    output_chart_files = getattr(args, "output_chart_file", [])
    output_csv_file = getattr(args, "output_csv_file", None)
    assert (
        output_chart_files or output_csv_file
    ), "At least one of --output-chart-file or --output-csv-file must be specified"

    with open(args.spec_file, encoding="utf-8") as spec_file:
        spec = json.load(spec_file)
    nominal_fps = spec["fps"]["num"] / spec["fps"]["den"]
    transition_count = spec["transition_count"]
    print(
        f"Successfully loaded spec file containing {transition_count} frame transitions"
        f" at {nominal_fps} FPS",
        file=sys.stderr,
    )

    transitions = pd.read_csv(
        args.edges_csv_file,
        usecols=["recording_timestamp_seconds", "edge_is_rising"],
    ).rename_axis(index="transition_index")
    transition_count = transitions.index.size
    transitions_interval_seconds = _interval(transitions.recording_timestamp_seconds)
    print(
        f"Recording analysis contains {transition_count} frame transitions, with first"
        f" transition at ~{transitions_interval_seconds.left:.6f} seconds and last"
        f" transition at ~{transitions_interval_seconds.right:.6f} seconds for a total"
        f" of ~{transitions_interval_seconds.length:.6f} seconds",
        file=sys.stderr,
    )

    transitions["time_since_previous_transition_seconds"] = (
        transitions.recording_timestamp_seconds.diff()
    )

    transitions["valid"] = transitions.edge_is_rising.pipe(
        lambda r: np.diff(r.values, prepend=not r.values[0])
    )
    invalid_transition_count = (~transitions.valid).sum()
    if invalid_transition_count > 0:
        print(
            f"WARNING: data contains {invalid_transition_count} edges where the"
            " previous edge is the same direction (i.e. transition from one color to"
            " the same color). This usually means the analyzer failed to make sense of"
            ' some of the recording. These transitions will be reported as "invalid".',
            file=sys.stderr,
        )

    high_is_white = None
    intentionally_delayed_transitions = spec["delayed_transitions"]
    if intentionally_delayed_transitions:
        delayed_transitions = _match_delayed_transitions(
            transitions,
            intentionally_delayed_transitions,
            transition_count,
            args.delayed_transition_max_offset,
        )

        if not delayed_transitions.empty:
            delayed_transitions = pd.concat(
                [transitions, delayed_transitions],
                axis="columns",
                join="inner",
            )
            # Since we know the expected transition indexes of the delayed
            # transitions, we can use them to deduce whether rising edges are
            # transitions to black or transitions to white.
            high_is_white = _is_high_white(delayed_transitions)
            if high_is_white is None:
                print(
                    "Unable to determine frame color information from delayed"
                    " transitions",
                    file=sys.stderr,
                )
            else:
                print(
                    "Deduced from delayed transitions that rising edges are"
                    f" transitions to {'white' if high_is_white else 'black'} and"
                    " falling edges are transitions to"
                    f" {'black' if high_is_white else 'white'}",
                    file=sys.stderr,
                )
            transitions["intentionally_delayed"] = pd.notna(
                delayed_transitions.expected_transition_index.reindex_like(transitions)
            )

    keep_first_transition = getattr(args, "keep_first_transition", False)
    if not keep_first_transition:
        transitions = transitions.iloc[1:]
        transitions.time_since_previous_transition_seconds.iloc[0] = np.nan
    keep_last_transition = getattr(args, "keep_last_transition", False)
    if not keep_last_transition:
        transitions = transitions.iloc[:-1]

    normal_transition = transitions.valid
    if "intentionally_delayed" in transitions:
        normal_transition = normal_transition & ~transitions.intentionally_delayed

    if getattr(args, "edge_direction_compensation", intentionally_delayed_transitions):
        falling_edge_lag_seconds = _estimate_falling_edge_lag_seconds(
            transitions[normal_transition]
        )
        falling_edge_offset_seconds = -falling_edge_lag_seconds / 2
        rising_edge_offset_seconds = falling_edge_lag_seconds / 2
        edge_direction_compensation_fineprint = (
            "Time since previous transition includes"
            f" {_si_format_plus(falling_edge_offset_seconds, 3)}s correction in all"
            f" falling edges and {_si_format_plus(rising_edge_offset_seconds, 3)}s"
            " correction in all rising edges"
        )
        transitions.loc[
            ~transitions.edge_is_rising, "time_since_previous_transition_seconds"
        ] += falling_edge_offset_seconds
        transitions.loc[
            transitions.edge_is_rising, "time_since_previous_transition_seconds"
        ] += rising_edge_offset_seconds
    else:
        edge_direction_compensation_fineprint = (
            "Consistent timing differences between falling and rising edges (i.e."
            " between black vs. white transitions) have NOT been compensated for"
        )

    time_between_transitions_stddev_seconds = transitions[
        normal_transition
    ].time_since_previous_transition_seconds.std()
    print(
        "Valid, non-delayed transition interval standard deviation:"
        f" ~{time_between_transitions_stddev_seconds:.6f} seconds",
        file=sys.stderr,
    )

    rounded_transitions = transitions.copy()
    rounded_transitions.recording_timestamp_seconds = (
        rounded_transitions.recording_timestamp_seconds.round(
            args.time_precision_seconds_decimals
        )
    )
    rounded_transitions.time_since_previous_transition_seconds = (
        rounded_transitions.time_since_previous_transition_seconds.round(
            args.time_precision_seconds_decimals
        )
    )

    if output_csv_file:
        rounded_transitions.pipe(
            lambda t: (
                t
                if high_is_white is None
                else t.assign(
                    to_white=t.edge_is_rising if high_is_white else ~t.edge_is_rising
                )
            )
        ).to_csv(output_csv_file, index=False)
    if output_chart_files:
        normal_transitions = transitions[normal_transition]
        shortest_transition = normal_transitions.iloc[
            normal_transitions.time_since_previous_transition_seconds.argmin()
        ]
        shortest_transition_duration = (
            shortest_transition.time_since_previous_transition_seconds
        )
        shortest_transition_timestamp = shortest_transition.recording_timestamp_seconds
        longest_transition = normal_transitions.iloc[
            normal_transitions.time_since_previous_transition_seconds.argmax()
        ]
        longest_transition_duration = (
            longest_transition.time_since_previous_transition_seconds
        )
        longest_transition_timestamp = longest_transition.recording_timestamp_seconds
        mean_time_between_transitions = (
            normal_transitions.time_since_previous_transition_seconds.mean()
        )
        p05_duration = transitions[
            normal_transition
        ].time_since_previous_transition_seconds.quantile(0.005)
        p95_duration = transitions[
            normal_transition
        ].time_since_previous_transition_seconds.quantile(0.995)
        outliers_count = (
            np.abs(
                stats.zscore(
                    transitions[normal_transition].loc[
                        :, "time_since_previous_transition_seconds"
                    ],
                    nan_policy="omit",
                )
            )
            > 3
        ).sum()
        mean_fps = 1 / mean_time_between_transitions
        found_intentionally_delayed_transitions = (
            transitions.intentionally_delayed.sum()
            if "intentionally_delayed" in transitions
            else 0
        )
        chart = _generate_chart(
            rounded_transitions,
            f"{transitions.index.size} transitions at {nominal_fps:.3f} nominal FPS",
            transitions_interval_seconds.left,
            high_is_white,
            args.chart_minimum_time_between_transitions_seconds,
            args.chart_maximum_time_between_transitions_seconds,
            np.round(
                mean_time_between_transitions, args.time_precision_seconds_decimals
            ),
            fine_print=[
                (
                    "First transition recorded at"
                    f" {si_format(transitions_interval_seconds.left, 3)}s; last:"
                    f" {si_format(transitions_interval_seconds.right, 3)}s; length:"
                    f" {si_format(transitions_interval_seconds.length, 3)}s"
                ),
                (
                    f"Detected {transition_count} transitions (expected"
                    f" {spec['transition_count']}); first transition was"
                    f" {'kept' if keep_first_transition else 'removed'}; last"
                    f" transition was {'kept' if keep_last_transition else 'removed'};"
                    f" expecting {len(intentionally_delayed_transitions)} intentionally"
                    " delayed transitions"
                ),
                (
                    f"The following stats exclude {invalid_transition_count} invalid"
                    " transitions and the"
                    f" {found_intentionally_delayed_transitions} intentionally delayed"
                    " transitions that were found:"
                ),
                edge_direction_compensation_fineprint,
                (
                    "Transition interval range:"
                    f" {si_format(shortest_transition_duration, 3)}s (at"
                    f" {si_format(shortest_transition_timestamp, 3)}s) to"
                    f" {si_format(longest_transition_duration, 3)}s (at"
                    f" {si_format(longest_transition_timestamp, 3)}s) - standard"
                    " deviation:"
                    f" {si_format(time_between_transitions_stddev_seconds, 3)}s"
                    f" - 99% of transitions are between {si_format(p05_duration, 3)}s"
                    f" and {si_format(p95_duration, 3)}s"
                ),
                (
                    "Mean time between transitions:"
                    f" {si_format(mean_time_between_transitions, 3)}s, i.e."
                    f" {mean_fps:.06f} FPS, which is {mean_fps/nominal_fps:.6f}x faster"
                    " than expected (clock skew)"
                ),
                (
                    f"{outliers_count} transitions are outliers (more than 3 standard"
                    " deviations away from the mean)"
                ),
                "Generated by videojitter",
            ],
        )
        for output_chart_file in output_chart_files:
            chart.save(output_chart_file)


if __name__ == "__main__":
    sys.exit(main())
