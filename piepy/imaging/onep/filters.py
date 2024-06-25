import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import freqz


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band", analog=False)

    return b, a


def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="low", analog=False)

    return b, a


def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    cutoff = cutoff / nyq
    b, a = butter(order, cutoff, btype="high", analog=False)

    return b, a


def butter_filter(signal, filter_type, cutoff, fs, order=3, plot=True):

    # filter signal
    signal_t = np.linspace(0, len(signal) * 1 / 10, len(signal))
    if filter_type == "low":
        if hasattr(cutoff, "__len__"):
            raise TypeError("Cutoff needs to be a scalar value for lowpass")
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, signal)
    elif filter_type == "high":
        if hasattr(cutoff, "__len__"):
            raise TypeError("Cutoff needs to be a scalar value for highpass")
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, signal)
    elif filter_type == "band":
        if not hasattr(cutoff, "__len__"):
            raise TypeError(
                "Cutoff needs to be [cutoff_low,cutoff_high] value for bandpass"
            )
        b, a = butter_bandpass(cutoff[0], cutoff[1], fs, order=order)
        y = filtfilt(b, a, signal)

    w, h = freqz(b, a, worN=2000)

    if plot:
        f, axs = plt.subplots(1, 2, figsize=(12, 8))
        axs[0].plot((fs * 0.5 / np.pi) * w, abs(h))

        if hasattr(cutoff, "__len__"):
            axs[0].axvline(cutoff[0], color="k", linestyle="--")
            axs[0].axvline(cutoff[1], color="k", linestyle="--")
            axs[0].plot(cutoff[0], 0.5 * np.sqrt(2), "ko")
            axs[0].plot(cutoff[1], 0.5 * np.sqrt(2), "ko")

        else:
            axs[0].plot(cutoff, 0.5 * np.sqrt(2), "ko")
            axs[0].axvline(cutoff, color="k", linestyle="--")
        axs[0].set_ylabel("Gain")
        axs[0].set_xlabel("Frequency (Hz)")
        axs[0].grid(True)
        axs[0].set_xlim([0, 1])

        axs[1].plot(signal_t, signal, label="Signal")
        axs[1].plot(signal_t, y, label="Filtered Signal")
        axs[1].set_xlabel("Time (seconds) ")
        axs[1].legend(loc="upper right")

    return y
