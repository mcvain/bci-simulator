import pandas as pd


def spatial_filter(eeg, channelSelection):
    # eeg = eeg.T
    eeg = (channelSelection).dot(eeg)
    eeg = eeg.T
    eeg = pd.DataFrame(eeg)

    return eeg