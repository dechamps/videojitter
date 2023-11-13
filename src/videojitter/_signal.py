"""Utilities for dealing with discrete-time signals."""

from collections import namedtuple
import soundfile
import scipy.signal

Signal = namedtuple("Signal", ["samples", "sample_rate"])
Signal.__doc__ = """Represents a discrete-time signal sampled whose values stored in the
`samples` NumPy array are regularly sampled at the frequency given by `sample_rate` (in
Hertz).
Signal objects are useful to ensure sample rate metadata is always passed around with
the sample values."""


def duration(signal):
    """Returns the duration of `signal` in seconds."""
    return signal.samples.size / signal.sample_rate


def downsample(signal, ratio, **kwargs):
    """Downsamples the signal by the given integer ratio. Additional arguments are
    passed to `scipy.signal.resample_poly()`."""
    return Signal(
        samples=scipy.signal.resample_poly(signal.samples, up=1, down=ratio, **kwargs),
        sample_rate=signal.sample_rate / ratio,
    )


def upsample(signal, ratio, **kwargs):
    """Upsamples the signal by the given integer ratio. Additional arguments are
    passed to `scipy.signal.resample_poly()`."""
    return Signal(
        samples=scipy.signal.resample_poly(signal.samples, up=ratio, down=1, **kwargs),
        sample_rate=signal.sample_rate * ratio,
    )


def butter(signal, **kwargs):
    """Filters the signal using a Butterworth IIR filter. Additional arguments are
    passed to `scipy.signal.butter()`."""
    return signal._replace(
        samples=scipy.signal.sosfilt(
            scipy.signal.butter(
                fs=signal.sample_rate,
                output="sos",
                **kwargs,
            ),
            signal.samples,
        )
    )


def firwin(sample_rate, pass_zero=True, **kwargs):
    """Equivalent to scipy.signal.firwin() but with a workaround for the
    `pass_zero=False` bug described at
    https://github.com/scipy/scipy/issues/19291.
    """
    kernel = scipy.signal.firwin(fs=sample_rate, **kwargs)
    if not pass_zero:
        kernel = -kernel
        kernel[kernel.size // 2] += 1
    return Signal(samples=kernel, sample_rate=sample_rate)


def correlate(signal1, signal2, *kargs, **kwargs):
    """Sample-rate-aware equivalent of `scipy.signal.correlate()`."""
    assert signal1.sample_rate == signal2.sample_rate
    return Signal(
        samples=scipy.signal.correlate(
            signal1.samples, signal2.samples, *kargs, **kwargs
        ),
        sample_rate=signal1.sample_rate,
    )


def convolve(signal1, signal2, *kargs, **kwargs):
    """Sample-rate-aware equivalent of `scipy.signal.convolve()`."""
    assert signal1.sample_rate == signal2.sample_rate
    return Signal(
        samples=scipy.signal.convolve(
            signal1.samples, signal2.samples, *kargs, **kwargs
        ),
        sample_rate=signal1.sample_rate,
    )


def fromfile(*kargs, **kwargs):
    """Loads a mono signal using `soundfile.read()`."""
    samples, sample_rate = soundfile.read(*kargs, **kwargs)
    assert samples.ndim == 1, (
        f"Recording file contains {samples.shape[1]} channels - only mono files"
        " are supported. Extract the correct channel and try again."
    )
    return Signal(samples=samples, sample_rate=sample_rate)


def tofile(signal, **kwargs):
    """Saves a signal using `soundfile.read()`."""
    return soundfile.write(
        data=signal.samples, samplerate=int(signal.sample_rate), **kwargs
    )
