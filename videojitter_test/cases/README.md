# videojitter test cases

For more details on how the videojitter test suite works, see [its README][].

Each subdirectory is a test case that checks a specific input and videojitter
commands/options against specific output goldens.

Each test case falls under one of the following:

## Synthetic inputs

### Fake recording generation and analysis

These test cases use `videojitter-generate-fake-recording` to generate a
synthetic recording. The fake recording is then analyzed and a report is
generated.

- `fake`
  - Uses the default settings of the fake recording generator.
- `high_amplitude`
  - Simulates a clipped recording.
- `ideal`
  - Uses fake recording generator settings that result in the cleanest, purest,
    simplest recording possible. This provides the easiest possible signal that
    the analyzer could possibly be asked to handle.
- `integer`
  - Generates a recording encoded with integer samples (instead of floating
    point).
- `nopadding`
  - Generates a recording that is aggressively trimmed with no padding before or
    after the test signal itself.

### Report generation

These test cases merely run the videojitter report generator directly on made up
inputs. No recording is analyzed.

- `bad_first_last_transition`
- `delayed_transitions`
- `edge_direction_compensation`
- `generate_report`
- `invalid_transitions`

### Video generation

The `generate_video` test case generates a test video using default settings.

Note that the test case merely checks that the video generator doesn't crash;
**the resulting video is not a test golden**, meaning the output itself isn't
checked. Using the generated video as a test golden would be technically
challenging due to the many potential sources of nondeterminism in the encoding
process.

## Real inputs

This section includes all test cases where a real physical display was measured
using a real physical instrument.

All recordings were made using the same instrument: a [Panasonic AMS3][] light
sensor directly attached to an ASUS Xonar U3 ADC.

The real test cases follow one of two naming conventions:

- `<SOURCE>_<FRAME RATE>p`
  - Indicates that the source is a hardware video player playing a test video
    encoded at the specified frame rate (FPS).
- `<SOURCE>_<FRAME RATE>p_at_<REFRESH RATE>`
  - Indicates that the source is a PC playing a test video encoded at the
    specified frame rate (FPS) to a display configured with the specified
    refresh rate (in Hz).

For example, `asusvlc_60p_at_240hz` means that the recording is of the "asusvlc"
source playing a 60 FPS test video at 240 Hz.

The test suite currently includes the following sources:

- `asusvlc`
  - The test video is being played on the built-in Nebula HDR display of an
    [ASUS ROG Flow X16 2023][] GV601VI-NL003W laptop running VLC on Windows 11.
- `asuswmp`
  - Same as above, except the built-in Windows Media Player is used to play the
    video.
- `lg`
  - The test video is being played on an LG OLED G1 TV using its built-in video
    file player.
- `evr`, `madvr`
  - Same as above, but the test video is being played by a Windows 11 PC using
    [EVR][] or [madVR][], respectively, and then sent over to the TV using the
    HDMI 2.1 output of an NVidia RTX 3080 Ti.
- `pixel5vlc`:
  - The test video is being played on the built-in display of a Google Pixel 5
    phone using the VLC Android app.

[ASUS ROG Flow X16 2023]:
  https://rog.asus.com/uk/laptops/rog-flow/rog-flow-x16-2023-series/
[EVR]:
  https://learn.microsoft.com/en-us/windows/win32/medfound/enhanced-video-renderer
[madvr]: https://forum.doom9.org/showthread.php?t=146228
[its README]: ../README.md
[Panasonic AMS3]:
  https://industrial.panasonic.com/cdbs/www-data/pdf/ADD8000/ADD8000C6.pdf
