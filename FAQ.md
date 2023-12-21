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
