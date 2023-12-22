# videojitter FAQ

## What is the purpose of the "warmup" and "cooldown" patterns?

By default, videojitter test videos include a 5-second "warmup" section and a
5-second "cooldown" section before and after the test signal itself. These serve
multiple purposes:

1. To ensure the boundaries of the test signal are clear to the analyzer.
   - Otherwise the analyzer could get confused and include some random activity
     before/after the test signal in the results.
2. To provide some padding, allowing the video playback system to "settle down"
   before the actual measurement starts.
   - For example, this gives time for potential video player On-Screen Display
     (OSD) elements to fade out so that they don't interfere with the
     measurement.
   - Some video playback systems also tend to have some benign frame timing
     issues at the very beginning of playback that are not representative of
     steady state operation.
3. To ensure that the average light level right before and after the test signal
   is roughly similar to the average light level of the test signal.
   - This is achieved using a blinking checkers pattern, which ensures that
     every pixel of the display sees the same on/off cycle as during the test
     signal.
   - The use of a checkers pattern, as opposed to full screen black/white,
     results in the overall pattern looking like "grey" to the instrument,
     assuming that it is located far enough away from the display for the light
     from the checker pattern to spatially integrate in the instrument. This
     ensures the analyzer won't mistake the pattern for the test signal.
     - One alternative could be to actually display a full screen grey color,
       but this is harder than it sounds because the correct color value would
       depend on gamma and on the effective duty cycle of the test signal on/off
       light output. The checkers pattern approach is more robust, though it is
       by no means perfect (e.g. it will fail to match the response of the test
       signal if the display uses "dynamic contrast").
   - The overarching goal is to prevent large, sudden average light level shifts
     at the beginning and end of the test signal, which would otherwise result
     in measurement artefacts especially with slow display/instruments and/or
     AC-coupled instruments.

You can adjust the length of these patterns, or remove them entirely, using the
`videojitter-generate-video` `--begin-padding` and `--end-padding` options. This
is especially useful if you want to measure the very beginning of playback. Of
course, if you remove the padding, you will lose the aforementioned benefits,
but the analyzer will do its best to make sense of your recording regardless.

## What is "edge direction compensation"?

Edge direction compensation is a feature of the videojitter report generator
that is aimed at suppressing distracting timing differences between transitions
to white frames vs. transitions to black frames.

To understand why this is useful, it is important to realize that displays tend
to exhibit different response curves (as in, light signal waveforms) when
transitioning to white vs. transitioning to black. This can lead to videojitter
measuring a consistent, systematic delay depending on the color of the frame
being transitioned to.

To be clear, this is not a measurement artefact, _per se_ - after all, these
delays are, in fact, induced by the video playback system, and one could argue
they should be reported as such. (This assumes the instrument itself does not
come with its own asymmetric response. It is unclear if this assumption holds in
practice.)

However, one could just as easily argue that these delays are misleading,
because they will tend to suggest that there is systematic timing error between
adjacent pairs of frames, but in reality this timing error is completely
dependent on the source and destination color of the pixel being transitioned.
In extreme cases this can lead to reports that suggest the presence of a 3:2
"24p@60Hz" pattern that isn't real, for example.

It is important to keep in mind that videojitter's goal is to measure _when_
transitions occur and how much time elapses _between_ them, not how individual
transitions look like or how long they take. If you want to do that, you'd
likely want to research the methods display hardware reviewers use to [measure
display pixel response][]. videojitter is more suitable for diagnosing issues
upstream of the display itself, where frames are not being sent to the display
at the right time.

For these reasons, videojitter will attempt to hide delays that uniformly affect
transitions to frames of the same color. This is what "edge direction
compensation" does.

Edge direction compensation is implemented by calculating the average delay in
rising edges (transitions to a given color) and separately for falling edges
(transitions to the opposite color). videojitter then uniformly applies a timing
correction to all same-direction edges throughout the entire recording so that
the difference disappears. The amount of correction applied is indicated in the
"fine print" (the text below the chart).

Edge direction compensation can be disabled by passing the
`--no-edge-direction-compensation` option to `videojitter-generate-report`. This
will likely result in the chart showing two separate "lines" as the transition
interval changes back and forth with every frame.

## What is the purpose of the "intentionally delayed transition"?

If you look closely at the test video that videojitter generates, you will
notice that there is a single frame exactly halfway through that is repeated,
i.e. the same color is shown for two frames in a row. This is not a bug;
videojitter duplicates that frame on purpose to create an _intentionally delayed
transition_.

This is done for two reasons:

1. To determine the polarity of the recording.
   - Intuitively, you'd expect a higher light level to translate to a higher
     signal level in the recording, but that is not always the case; with some
     instruments it's the opposite.
   - For user convenience, videojitter attempts to autodetect the polarity of
     the recording. The way this works is, videojitter looks at the recording
     level for the repeated frame, and since it knows what color the repeated
     frame is, it can deduce the mapping between signal level and frame color.
   - If the autodetection succeeds, videojitter will report frame colors (i.e.
     "black" or "white"); if it fails, the chart will only show edge direction
     (i.e. "falling edge" or "rising edge").
2. To break up any patterns that affect the relative timing of successive pairs
   of transitions.
   - The textbook example is a 3:2 "24p@60Hz" pattern, where every other frame
     is displayed for a different amount of time.
   - If there was no delayed transition, the pattern would _de facto_ induce a
     systematic delay in white frames (or black frames).
   - This is a problem because it would interact badly with [edge direction
     compensation][]: indeed videojitter would be unable to tell whether the
     difference in duration between black frames and white frames is due to
     benign differences in display pixel response (which we don't care about),
     or whether it's caused by actual differences in frame presentation timing
     such as a 3:2 pattern (which we definitely do want to know about).
   - The goal of the delayed transition is to invert that relationship,
     decorrelating frame color from any frame timing pattern.

During report generation, videojitter will attempt to automatically locate the
intentionally delayed transition among the other transitions in the recording
and annotate it accordingly. Note this autodetection relies on heuristics (it's
basically looking for a transition that "stands out" from its neighbors around
the timestamp where the delayed transition is supposed to be), which means it
doesn't always work. If autodetection fails, the delayed transition will not be
annotated on the chart and frame colors won't be shown. It's also possible for
videojitter to [incorrectly locate the delayed transition][], which means the
annotation will be on the wrong transition, and the frame color information may
be inverted as well.

You can omit the delayed transition from the test video (and subsequent
analysis) by passing the `--no-delayed-transition` option to
`videojitter-generate-spec`. Note that this will prevent videojitter from
determining frame colors, and it will also disable [edge direction
compensation][] by default to avoid potentially misleading results.

[edge direction compensation]: #what-is-edge-direction-compensation
[measure display pixel response]:
  https://tftcentral.co.uk/articles/response_time_testing
[incorrectly locate the delayed transition]:
  EXAMPLES.md#artefacts-caused-by-overly-slow-displayinstrument
