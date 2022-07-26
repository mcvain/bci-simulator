from spectrum import arma2psd, arburg
from pylab import log10, pi
from math import floor, ceil, isnan
import numpy as np


def decoder_arpsd(eeg, bufferLength, x_control_buffer, y_control_buffer, normalize_mode):
    modelOrder = 16
    spectralDensity_output = []
    for i in range(0, len(eeg.columns)):
        # channel = str(i)
        EEGdata_ch = eeg[i].tolist()
        AR, P, k = arburg(EEGdata_ch, modelOrder)
        PSD = arma2psd(AR)
        PSD = PSD[len(PSD):len(PSD) // 2:-1]  # takes 2-sided PSD to 1-sided
        PSD = 10 * log10(abs(PSD) * 2. / (2. * pi))
        spectralDensity_Burg = PSD[floor(2047 / 256 * 8):ceil(2047 / 256 * 13):1]  # PSD from 8 to 13 Hz.
        spectralDensity_output.append(np.mean(spectralDensity_Burg))  # alpha power

    if normalize_mode == 0:  # operate then normalize
        # uncorrected
        x_control_raw = spectralDensity_output[1] - spectralDensity_output[0]
        y_control_raw = - spectralDensity_output[1] - spectralDensity_output[0]

        x_control = (x_control_raw - np.mean(x_control_buffer[-bufferLength:])) / (
            np.std(x_control_buffer[-bufferLength:]))
        y_control = (y_control_raw - np.mean(y_control_buffer[-bufferLength:])) / (
            np.std(y_control_buffer[-bufferLength:]))

        # save into buffer for time-history
        x_control_buffer.append(x_control_raw)
        y_control_buffer.append(y_control_raw)  # corrected into time history is closer to BCI2000 behavior

    elif normalize_mode == 1:  # normalize and then operate
        # save into buffer for time-history
        x_control_buffer.append(spectralDensity_output[0])
        y_control_buffer.append(spectralDensity_output[1])

        C3_normalized = (spectralDensity_output[0] - np.mean(x_control_buffer[-bufferLength:])) / np.std(
            x_control_buffer[-bufferLength:])
        C4_normalized = (spectralDensity_output[1] - np.mean(y_control_buffer[-bufferLength:])) / np.std(
            y_control_buffer[-bufferLength:])

        x_control = 1 * C4_normalized + (-1) * C3_normalized
        y_control = -1 * C4_normalized - 1 * C3_normalized

    # when buffer only has 1 value at the start
    if isnan(x_control):
        x_control = 0
    if isnan(y_control):
        y_control = 0

    return x_control, y_control, x_control_buffer, y_control_buffer
