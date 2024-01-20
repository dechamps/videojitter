# videojitter test cases

For more details on how the videojitter test suite works, see [its README][].

Each subdirectory is a test case that checks a specific input and videojitter
commands/options against specific output goldens.

**Note:** in some cases, report chart _images_ are treated as test goldens and
included in version control. This is usually because the images are used as
examples in videojitter documentation - this mechanism is a convenient way to
ensure the images always stay in sync with any changes to videojitter outputs.

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

Unless otherwise noted, all recordings were made using the same instrument: a
[Panasonic AMS3][] light sensor directly attached to the microphone jack of an
Asus Xonar U3 audio interface. That microphone jack exposes an unloaded DC bias
voltage of ~1.9 V.

The real test cases follow one of two naming conventions:

- `<SOURCE>_<FRAME RATE>p`
  - Indicates that the source is a hardware video player playing a test video
    encoded at the specified frame rate (FPS).
- `<SOURCE>_<FRAME RATE>p_inst_<MANUFACTURER>_<MODEL>_<ADC>`
  - Same as above, but measured with a specific instrument.
  - The name includes the manufacturer and model of the specific light sensor
    used.
  - The last part indicates which [ADC][] setup is used. The test suite
    currently includes 3 possible ADC setups:
    - `qa401`: the light sensor is directly connected to one of the inputs of a
      [QuantAsylum QA401][] audio analyzer. There is no DC bias, so photodiodes
      operate in forward (photovoltaic) mode.
    - `u3f`: the light sensor is directly connected to the microphone jack of an
      Asus Xonar U3. The cathode (-) of the light sensor is connected to the
      jack sleeve, i.e. ground, such that there is a positive DC bias voltage on
      the anode (+). This makes photodiodes operate in forward voltage mode with
      a DC bias added on top.
    - `u3r`: same as above, but the polarity is reversed, such that photodiodes
      operate in reverse voltage (photoconductive) mode using the DC bias
      voltage.
- `<SOURCE>_<FRAME RATE>p_at_<REFRESH RATE>`
  - Indicates that the source is a PC playing a test video encoded at the
    specified frame rate (FPS) to a display configured with the specified
    refresh rate (in Hz).
- `<SOURCE>_<FRAME RATE>p_vrr`
  - Indicates that the source is a PC playing a test video encoded at the
    specified frame rate (FPS) using [Variable Refresh Rate (VRR)][].

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
- `evr`, `madvr`, `mpv`
  - Same as above, but the test video is being played by a Windows 11 PC using
    [EVR][], [madVR][] or [mpv][], respectively, and then sent over to the TV
    using the HDMI 2.1 output of an NVidia RTX 3080 Ti.
  - In the case of `mpv`:
    - The measurements were made using a [OSRAM SFH 213][] connected to a
      [QuantAsylum QA401][] audio analyzer.
    - `mpv.log` contains the output of [`--log-file`][].
- `pixel5vlc`:
  - The test video is being played on the built-in display of a Google Pixel 5
    phone using the VLC Android app.

[ADC]: https://en.wikipedia.org/wiki/Analog-to-digital_converter
[ASUS ROG Flow X16 2023]:
  https://rog.asus.com/uk/laptops/rog-flow/rog-flow-x16-2023-series/
[EVR]:
  https://learn.microsoft.com/en-us/windows/win32/medfound/enhanced-video-renderer
[`--log-file`]: https://mpv.io/manual/stable/#options-log-file
[madvr]: https://forum.doom9.org/showthread.php?t=146228
[mpv]: https://mpv.io/
[its README]: ../README.md
[OSRAM SFH 213]:
  https://ams-osram.com/products/photodetectors/photodiodes/osram-radial-t1-34-sfh-213
[Panasonic AMS3]:
  https://industrial.panasonic.com/cdbs/www-data/pdf/ADD8000/ADD8000C6.pdf
[QuantAsylum QA401]: https://quantasylum.com/products/qa401-audio-analyzer
[Variable Refresh Rate (VRR)]:
  https://en.wikipedia.org/wiki/Variable_refresh_rate
