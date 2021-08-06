from spectrum import aryule, Periodogram, arma2psd, arburg, pyule
from pylab import log10, pi
from math import floor, ceil, isnan
import numpy as np


def decoder_arpsd(eeg, x_control_buffer, y_control_buffer, normalize_mode):
    modelOrder = 16
    bufferLength = 900
    spectralDensity_output_Burg = []
    for i in range(0, len(eeg.columns)):
        # channel = str(i)
        EEGdata_ch = eeg[i].tolist()
        AR, P, k = arburg(EEGdata_ch, modelOrder)
        PSD = arma2psd(AR)
        PSD = PSD[len(PSD):len(PSD) // 2:-1]  # takes 2-sided PSD to 1-sided (something like https://www.physik.uzh.ch/local/teaching/SPI301/LV-2015-Help/lvanlsconcepts.chm/lvac_convert_twosided_power_spec_to_singlesided.html)
        PSD = 10 * log10(abs(PSD) * 2. / (2. * pi))
        spectralDensity_Burg = PSD[floor(2047 / 256 * 8):ceil(2047 / 256 * 12):1]  # PSD from 8 to 12 Hz.  # 80 is sampling rate/2.
        spectralDensity_output_Burg.append(np.mean(spectralDensity_Burg))  # alpha power

    if normalize_mode == 0:  # operate then normalize
        y_control_raw = spectralDensity_output_Burg[0] + spectralDensity_output_Burg[1]  # C3 + C4
        x_control_raw = spectralDensity_output_Burg[1] - spectralDensity_output_Burg[0]  # C4 - C3

        # Save into buffer for time-history
        y_control_buffer.append(y_control_raw)
        x_control_buffer.append(x_control_raw)

        # How long should the buffer length be?
        x_control = (x_control_raw - np.mean(x_control_buffer[-bufferLength:])) / (
            np.std(x_control_buffer[-bufferLength:]))
        y_control = (y_control_raw - np.mean(y_control_buffer[-bufferLength:])) / (
            np.std(y_control_buffer[-bufferLength:]))

        d = 0.3

        # Dwell state implementation
        if abs(x_control) >= d:
            x_control = x_control - d * np.sign(x_control)
        elif abs(x_control) < d:
            x_control = x_control / 10
            # del x_control_buffer[-1]
            x_control_buffer.append(x_control)

        if abs(y_control) >= d:
            y_control = y_control - d * np.sign(y_control)
        elif abs(y_control) < d:
            y_control = y_control / 10
            # del y_control_buffer[-1]
            y_control_buffer.append(y_control)

    elif normalize_mode == 1:  # normalize and then operate
        # Save into buffer for time-history
        x_control_buffer.append(spectralDensity_output_Burg[0])
        y_control_buffer.append(spectralDensity_output_Burg[1])

        C3_normalized = (spectralDensity_output_Burg[0] - np.mean(x_control_buffer[-bufferLength:])) / np.std(
            x_control_buffer[-bufferLength:])
        C4_normalized = (spectralDensity_output_Burg[1] - np.mean(y_control_buffer[-bufferLength:])) / np.std(
            y_control_buffer[-bufferLength:])
        x_control = (C4_normalized - C3_normalized)
        y_control = (C3_normalized + C4_normalized)

        d = 0.3

        # Dwell state implementation
        # if abs(x_control) >= d:
        #     x_control = x_control - d * np.sign(x_control)
        # elif abs(x_control) < d:
        #     x_control = x_control / 10
        # if abs(y_control) >= d:
        #     y_control = y_control - d * np.sign(y_control)
        # elif abs(y_control) < d:
        #     y_control = y_control / 10

    # When buffer only has 1 value at the start, combat nan problem.
    if isnan(x_control):
        x_control = 0
    if isnan(y_control):
        y_control = 0

    return x_control, y_control, x_control_buffer, y_control_buffer, d
