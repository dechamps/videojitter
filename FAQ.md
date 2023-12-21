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

Edge direction compensation is implemented by looking at the average apparent
frame duration for white frames, and separately for black frames. videojitter
then uniformly applies a timing correction to all frames of the same color
throughout the entire recording so that the difference disappears. The amount of
correction applied is indicated in the "fine print" (the text below the chart).

Edge direction compensation can be disabled by passing the
`--no-edge-direction-compensation` option to `videojitter-generate-report`. This
will likely result in the chart showing two separate "lines" as the transition
interval changes back and forth with every frame.

[measure display pixel response]:
  https://tftcentral.co.uk/articles/response_time_testing
