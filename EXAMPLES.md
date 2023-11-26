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

## Systematic timing errors

<img src="videojitter_test/cases/lg_23p/test_output/report.svg">

The above result was obtained by playing a 24/1.001 FPS video on the built-in
player of an LG G1 OLED TV.

Just like the previous example we observe timing errors, but this time they are
not isolated. The typical error is ±8.3 ms, which suggests that the TV is
running at 120 Hz internally. The built-in video player seems unable to reliably
meet individual refresh targets and is constantly "missing the mark". This is a
surprisingly poor result for a built-in hardware video player.

While it may look like the errors follow a precise periodic pattern, this is not
quite the case - if we zoom in we see the behavior is not exactly the same each
time:

<img src="videojitter_test/cases/lg_23p/test_output/zoomed_report.svg">

## Periodic timing errors

<img src="videojitter_test/cases/asuswmp_23p_at_240hz/test_output/report.svg">

The above result was obtained by playing a 24/1.001 FPS video on the built-in
"Nebula HDR" display of an ASUS ROG Flow X16 2023 laptop using Windows Media
Player on Windows 11. The display is being driven at 240 Hz.

Aside from a couple of outliers near the 11-second mark, what is interesting in
this example is a seemingly periodic pattern of delayed frames. Each frame is
delayed by exactly one refresh interval, which at 240 Hz is ~4.2 ms.

This phenomenon can be explained by the display refresh rate (240 Hz) not being
a perfect whole multiple of the video FPS (24/1.001 FPS). Most of the time each
frame is displayed for exactly 10 refresh intervals, which is ~41.666666 ms, but
that's slightly wrong. In a 24/1.001 FPS video, each frame should be displayed
for slightly longer: ~41.708333 ms. As a result, an absolute, constant timing
error of ~0.041666 ms accumulates with each new frame. This goes on until the
error reaches the duration of a whole refresh interval (~4.16666) ms), which
happens after exactly 100 frames or ~4.2 seconds. At that point, the system
delays the next frame by one refresh interval so that the playback clock can
"catch up" to the excessively fast display refresh clock. If this hypothesis is
correct, we should expect to see a transition delayed by ~4.2 ms every ~4.2
seconds… and that is exactly what we see here!

This example also demonstrates that you don't need a fast instrument to
accurately measure high refresh rate displays: here the display refresh interval
is ~4.2 ms, but it was measured using an instrument that takes ~8.5 ms to
settle. Even then, the instrument was still able to very precisely measure the
timing errors shown above. This is because what really matters is not display
refresh rate, but the frame rate of the test video. In this example the test
video is 24/1.001 FPS which is well within the limits of the instrument.

## 3:2 "23p@60Hz" pattern

<img src="videojitter_test/cases/evr_23p_at_59hz/test_output/report.svg">

The above is a textbook example of a 24 FPS test video being played on a display
being driven at a 60 Hz refresh rate, leading to the infamous "3:2 pattern"
which is very commonplace when playing video on PC.

Basically, the problem is that 60 is not a whole multiple of 24, so the only way
for the video player to make that work is to display a frame for 2 refresh
intervals (2/60), then 3 refresh intervals (3/60), then 2, then 3, etc. This
leads to the correct _average_ overall frame duration of 5/120 or 1/24 so the
video can play at the proper speed, but at the cost of a very significant timing
error on each frame.

On a videojitter chart, this takes the form of a very obvious pattern where the
frame transition intervals arrange themselves into two clear "lines": one at ~3
3ms (2 60 Hz refresh intervals) and one at ~50 ms (3 60 Hz refresh intervals).

Zooming in, we can easily observe that frame durations indeed alternate between
the two from one frame to the next, as expected:

<img src="videojitter_test/cases/evr_23p_at_59hz/test_output/zoomed_report.svg">

The reason why the colours swap positions in the middle of the first chart is
because of the intentionally delayed transition, which causes the pattern
between black and white frames to be inverted.

## Severe presentation delays

<img src="videojitter_test/cases/pixel5vlc_59p/test_output/report.svg">

The above result was obtained by playing a 60/1.001 FPS video on a Google Pixel
5 smartphone using the Android VLC media player app.

While the video is being played correctly most of the time, one can easily spot
severe issues around the 10, 27 and 46 second marks. This is consistent with
what could be observed during measurement, where the playback system was visibly
struggling to the naked eye. The text below the chart also points out that
hundreds of transitions are missing, suggesting that the playback system dropped
many frames. It would appear this playback system is simply not capable of
keeping up with the frame rate of the test video.

This example demonstrates that, even when the system under test is badly
misbehaving, videojitter can still make sense of the data and produces reports
that accurately reflects what went right and what went wrong.

## Artefacts caused by overly slow display/instrument

<img src="videojitter_test/cases/asuswmp_240p_at_240hz/test_output/report.svg">

The above result was obtained by playing a 240 FPS video on the built-in "Nebula
HDR" display of an ASUS ROG Flow X16 2023 laptop using Windows Media Player on
Windows 11. The display is being driven at 240 Hz.

The instrument used to make this measurement has a documented response time of
8.5 ms, making it challenging to measure a 240 FPS video which has frame
intervals of ~4.2 ms. It is quite likely that even the display itself isn't fast
enough anyway: indeed, current-technology LCD displays are unlikely to be able
to complete a full black-white transition in such a short amount of time.

In this way the videojitter test video is arguably more demanding than
real-world content, which rarely flips between 0% black and 100% white in the
span of a single frame. Not to mention 240 FPS video is not exactly commonplace…

It is not possible, in general, to distinguish between a slow instrument and a
slow display in a videojitter measurement, as both would look roughly the same
in the raw recording. However, we can discuss the effects of a "slow recording"
as shown in the above example.

The chart _mostly_ looks right despite the equipment being driven to its limits.
videojitter will still attempt to extract what it can from the recording, and is
able to time the vast majority of transitions correctly.

However, the text at the bottom of the chart warns that 4 transitions went
"missing". Also, things look a bit weird when sudden changes in frame durations
occur, such as around the intentionally delayed transition and the outlier at
around 35 seconds. Let's zoom into the latter:

<img src="videojitter_test/cases/asuswmp_240p_at_240hz/test_output/zoomed_report.svg">

One transition happened ~1.7 ms after the previous one, something that the
display is not even capable of!

The problem with sudden changes in frame durations is that, if a frame is
displayed for longer, then the slow response time has less of an impact and the
display and instrument will be able to go further into the transition than they
otherwise could. This may sound like a good thing, but that actually poses a
challenge for the videojitter analyzer because this means the waveform of longer
transitions look different from shorter transitions - not just in duration, but
also in amplitude and shape. This makes it harder for videojitter to determine
where the true transition time is.

Consider, for example, how the intentionally delayed transition looks like in
the raw recording:

<img src="img/audacity-slow-delayed.png">

One can see that, because the delayed transition is a longer frame, the signal
peaks significantly higher than for the surrounding transitions. In this
situation, there is ambiguity (illustrated by the highlighted area) as to what
the precise transition time should be. Currently videojitter will tend to pick a
time closer to the right of the highlighted area, which in this case happens to
be wrong (but it might not be in other cases, which is what makes this problem
difficult to solve). This leads to videojitter interpreting the transition as
being excessively late. Consequently, the next transition is interpreted as
being excessively early - hence impossibly quick transitions are being reported.

This also leads to videojitter getting confused as to which frames are black and
which frames are white. Videojitter uses the delayed transition to deduce this,
as it knows what color the longer-lasting frame is supposed to be. Here the true
delayed transition is the falling edge after the large peak, but videojitter
thinks it's the (mistimed) rising edge before the peak. Therefore, videojitter
incorrectly deduces that rising edges are white and falling edges are black - in
reality it's the opposite in this particular measurement.

These issues can be more or less pronounced depending on the frame rate and
instrument/display speed. Here's another example where the same 240 Hz display
is being measured playing 60 Hz video where the player (VLC in this case) is
missing its target very frequently, leading to many random shifts in frame
durations:

<img src="videojitter_test/cases/asusvlc_60p_at_240hz/test_output/report.svg">

In that example we can see that videojitter is slightly mistiming most
transitions due to the sudden random shifts, which causes the lines on the chart
to look fuzzy/noisy instead of exactly matching a multiple of the display
refresh interval. Looking at the raw recording, it becomes clear why videojitter
struggles to precisely time these transitions, given their amplitude and shape
is all over the place:

<img src="img/audacity-asusvlc_60p_at_240hz.png">

The overall takeaway here is: while videojitter is still mostly usable when
measuring high frame rates using excessively slow displays and/or instruments,
it is important to watch out for these artefacts while interpreting
measurements. People making 24 FPS measurements can normally ignore these issues
as that frame rate should be well within the limits of any reasonable display
and instrument.

[build a similar instrument for yourself]: INSTRUMENT.md
[madVR]: https://forum.doom9.org/showthread.php?t=146228
