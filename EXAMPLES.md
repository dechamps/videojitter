# videojitter examples gallery

This page shows some videojitter measurements of real-world systems and a
discussion of the various phenomena they surface. It can be seen as a showcase
of what videojitter is capable of, or as a guide on how to interpret the various
patterns that sometimes appear in practical measurements.

All measurements shown here have been made using a cheap homemade instrument,
specifically a Panasonic AMS3 light sensor feeding an ASUS Xonar U3 used as an
ADC. You can [build a similar instrument for yourself][] in minutes.

## Perfect result

<img src="videojitter_test/cases/lg_119p_lowbrightness/test_output/report.svg">

This result, obtained by playing a 120/1.001 FPS video on the built-in player of
an LG G1 OLED TV, it the best result a videojitter user could possibly hope for.

The frame transitions form an extremely clear, sharp, straight horizontal line
on the chart with no deviations whatsoever, except for the intentionally delayed
transition (as expected). This means the time interval between frames (8.34 ms,
as expected for 120/1.001 FPS) stayed exactly the same throughout the entire
test signal with no outliers - no late nor early frames.

This example also demonstrates the ability of both the playback and measurement
system to handle very high FPS - much higher than typical video content - as
well as their amazing timing accuracy, which as described in the "fine print"
(the text below the chart) is in the order of 10 µs (yes, *micro*seconds).

If you get this kind of result, your playback system is basically flawless as
far as frame presentation timing is concerned.

## Isolated timing errors

<img src="videojitter_test/cases/madvr_23p_at_119hz/test_output/report.svg">

This result was obtained by playing a 24/1.001 FPS video (typical of most movie
content) using the [madVR][] software video player running on a Windows 11 PC.
The display device is the same as the previous example and is being driven at a
120/1.001 Hz refresh rate.

Compared to the perfect example above, we can spot a couple of differences.

The line is higher, which simply reflects the slower frame rate (longer
transitions between frames). It's worth noting that, even though the display is
running at 120 Hz, the test video is running at 24 FPS, so that's what
videojitter sees - obviously it cannot observe individual display refreshes if
the frame does not change.

More importantly, one immediately spots a few _unexpected_ outliers on the chart
(the intentionally delayed transition is also an outlier, but it's expected).
This means there are transitions that did not happen after the expected 41.71 ms
since the previous one: the frame was presented either too early (below the
line) or too late (above the line).

This is a typical result when playing videos using PC software players, which
suffer from a number of technical challenges that make it difficult for the
player to perfectly time frame presentation.

In this particular case, the delayed transition at about 6 seconds into the
recording was likely caused by the video output clock running too fast relative
to the playback clock, forcing the system to delay a frame to compensate.

The second pair of late and early transitions at about 25 seconds was likely
caused by the playback system "missing a beat" on a single frame (possibly due
to a thread scheduling deadline miss), and then showing a later frame earlier to
"catch up".

It's also worth noting the amount of time by which the transition interval
deviated on these early and late frames. It turns out that every one of these
outliers deviated by ±8.34 milliseconds. This is completely unsurprising given
the display refresh interval in this example was 120/1.001 Hz - the frame was
simply displayed one refresh interval too early or too late, as one would
naturally expect.

## Frequent timing errors

<img src="videojitter_test/cases/lg_23p/test_output/report.svg">

The above result was obtained by playing a 24/1.001 FPS video on the built-in
player of an LG G1 OLED TV.

Just like the previous example we observe timing errors, but this time they are
not isolated. The typical error is ±8.3 ms, which suggests that the TV is
running at 120 Hz internally. The built-in video player seems unable to reliably
meet individual refresh targets and is constantly "missing the mark". This is a
surprisingly poor result for a built-in hardware video player.

[build a similar instrument for yourself]: INSTRUMENT.md
[madVR]: https://forum.doom9.org/showthread.php?t=146228
