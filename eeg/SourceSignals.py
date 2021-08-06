# Functions to generate source activation signals
import numpy as np
import scipy.signal as signal
from typing import Tuple
import random
from scipy.io import loadmat
from numpy.random import default_rng
# from line_profiler_pycharm import profile
# from opt_einsum import contract

rng = default_rng(42)


def epoch_config(srate, epochLength):
    # srate = sampling rate
    # epochLength = length of an epoch in ms
    # samples = number of samples in an epoch (i.e. number of rows in EEG epoch)
    samples = int(np.floor(srate * epochLength / 1000))
    return srate, epochLength, samples


srate, epochLength, samples = epoch_config(srate=512, epochLength=100)


def import_leadfield(mat_name, mode):
    """
    Load leadfield matrix.

    Output:
    leadfield - projections in xyz directions (channels x sources x 3)
    orientation - orientation for each source
    pos - xyz coordinates of each source
    chanlocs - channel metadata
    """
    leadfield = loadmat(mat_name)
    if mode == 'SEREEGA' or mode == 'sereega':
        lf = leadfield['nyhead_lf_test'][0][0][0]
        orientation = leadfield['nyhead_lf_test'][0][0][1]  # default orientations (orthogonal)
        pos = leadfield['nyhead_lf_test'][0][0][2]
        chanlocs = leadfield['nyhead_lf_test'][0][0][3]

    elif mode == 'brainstorm' or mode == 'brainnetome':
        lf = leadfield['model'][0][0][4]
        orientation = leadfield['model'][0][0][8]
        pos = leadfield['model'][0][0][7]
        chanlocs = leadfield['model'][0][0][14]

    lf = lf.astype(np.float32)
    orientation = orientation.astype(np.float32)
    pos = pos.astype(np.float32)

    return leadfield, lf, orientation, pos, chanlocs


def import_lf_within_source_signals(leadfield_mode):
    global leadfield, lf, orientation, pos, chanlocs
    if leadfield_mode == 'sereega':
        leadfield, lf, orientation, pos, chanlocs = import_leadfield('data/leadfield/32_channel_nyhead_lf.mat', 'sereega')
    elif leadfield_mode == 'brainstorm':
        leadfield, lf, orientation, pos, chanlocs = import_leadfield('data/leadfield/openmeeg_fsaverage_lf.mat', 'brainstorm')
    return leadfield, lf, orientation, pos, chanlocs


def kaiserord(f: np.ndarray, a: np.ndarray, dev: np.ndarray, fs: float = 2) -> Tuple:
    """Kaiser window FIR filter design estimation parameters
    Parameters
    ----------
    f : ndarray
        Band edges. The length of `f` is the length of `2*len(a)-2`.
    a : ndarray
        Band amplitude. The amplitude is specified on the bands defined by `f`.
        Together, `f` and `a` define a piecewise-constant response function.
    dev : ndarray
        Maximum allowable deviation.
        `dev` is a vector the same size as `a` that specifies the maximum allowable
        deviation between the frequency response of the output filter and its band
        amplitude, for each band. The entries in dev specify the passband ripple
        and the stopband attenuation. Specify each entry in `dev` as a positive number,
        representing absolute filter gain (unit-less).
    fs : float, optional
        Sample rate in Hz. Use this syntax to specify band edges scaled to a particular
        application's sample rate. The frequency band edges in f must be from 0 to fs/2.
        Default is 2.

    Raises
    ------
    ValueError
        If the length of `f` is not same as `2*len(a)-2`.
        If the length `a` and `dev` is not the same.
        If `dev` includes minus value.

    Returns
    -------
    n : int
        Filter order.
    Wn : ndarray
        Normalized frequency band edges.
    beta : float
        The `beta` parameter to be used in the formula for a Kaiser window.
    ftype : string
        Filter type of filter('low', 'high', 'bandpass', 'stop', 'DC-0'
            or 'DC-1').
            Specified as one of the following.

            1. 'low' specifies a lowpass filter with cutoff frequency Wn.
               'low' is the default for scalar Wn.
            2. 'high' specifies a highpass filter with cutoff frequency Wn.
            3. 'bandpass' specifies a bandpass filter if Wn is a two-element vector.
               'bandpass' is the default when Wn has two elements.
            4. 'stop' specifies a bandstop filter if Wn is a two-element vector.
            5. 'DC-0' specifies that the first band of a multiband filter is
               a stopband.
               'DC-0' is the default when Wn has more than two elements.
            6. 'DC-1' specifies that the first band of a multiband filter is
               a passband.

    BSD 3-Clause License
    Copyright (c) 2019, Yuki Fukuda
    All rights reserved.
    """

    if type(f) != np.ndarray:
        if type(f) == list:
            f = np.array(f)
        else:
            f = np.array([f])

    if type(a) != np.ndarray:
        if type(a) == list:
            a = np.array(a)
        else:
            a = np.array([a])

    if type(dev) != np.ndarray:
        if type(dev) == list:
            dev = np.array(dev)
        else:
            dev = np.array([dev])

    # Parameter check
    if len(f) != 2 * len(a) - 2:
        raise ValueError("The length of 'f' must be the length of 2*len(a)-2.")

    if np.any(a[0:len(a) - 2] != a[2:len(a)]):
        raise ValueError("Pass and stop bands in a must be strictly alternating.")

    if (len(dev) != len(a)) and (len(dev) != 1):
        raise ValueError("'dev' and 'a' must be the same size.")

    dev = np.min(dev)
    if dev <= 0:
        raise ValueError("'dev' must be larger than 0.")

    # Calcurate normalized frequency band edges.
    Wn = (f[0:len(f):2] + f[1:len(f):2]) / fs

    # Determine ftype
    if len(Wn) == 1:
        if a[0] > a[1]:
            ftype = 'low'
        else:
            ftype = 'high'
    elif len(Wn) == 2:
        if a[0] > a[1]:
            ftype = 'stop'
        else:
            ftype = 'bandpass'
    else:
        if a[0] > a[1]:
            ftype = 'DC-1'
        else:
            ftype = 'DC-0'

    # Calcurate beta
    A = -20 * np.log10(dev)
    beta = signal.kaiser_beta(A)

    # Calcurate n from beta and dev
    width = 2 * np.pi * np.min(f[1:len(f):2] - f[0:len(f):2]) / fs
    n = np.max((1, int(np.ceil((A - 8) / (2.285 * width)))))

    # If last band is high, make sure the order of the filter is even
    if ((a[0] > a[1]) == (len(Wn) % 2 == 0)) and (n % 2 == 1):
        n += 1

    if len(Wn) == 1:
        Wn = Wn[0]

    return int(n), Wn, beta, ftype


def fir1(n: int, Wn, ftype: str = 'default', window='hamming', scaleopt: bool = True) -> Tuple:
    """
    FIR filter design using the window method.

    This function computes the coefficients of a finite impulse response filter.
    The filter will have linear phase; it will be Type I if n is odd
    and Type II if numtaps is even.
    Type II filters always have zero response at the Nyquist frequency,
    so a ValueError exception is raised if firwin is called with n even
    and having a passband whose right end is at the Nyquist frequency.


    Parameters
    ----------
        n : int
            Filter order.
            `n` must be even if a passband includes the Nyquist frequency.

        Wn : float or 1D array_like
            Cutoff frequency of filter (expressed in the same units as fs)
            OR an array of cutoff frequencies (that is, band edges).
            In the latter case, the frequencies in `Wn` should be positive
            and monotonically increasing between 0 and 1.
            The values 0 and 1 must not be included in `Wn`.

        ftype : string, optional
            Filter type of filter('low', 'high', 'bandpass', 'stop', 'DC-0'
            or 'DC-1').
            Specified as one of the following.

            1. 'low' specifies a lowpass filter with cutoff frequency Wn.
               'low' is the default for scalar Wn.
            2. 'high' specifies a highpass filter with cutoff frequency Wn.
            3. 'bandpass' specifies a bandpass filter if Wn is a two-element vector.
               'bandpass' is the default when Wn has two elements.
            4. 'stop' specifies a bandstop filter if Wn is a two-element vector.
            5. 'DC-0' specifies that the first band of a multiband filter is
               a stopband.
               'DC-0' is the default when Wn has more than two elements.
            6. 'DC-1' specifies that the first band of a multiband filter is
               a passband.

        window : string or tuple of string and parameter values, optional
            Desired window to use. See 'scipy.signal.get_window' for a list of
            windows and required parameters.

        scaleopt : bool, optional
            Set to True to scale the coefficients so that the frequency response
            is exactly unity at a certain frequency. That frequency is either:

            - 0 (DC) if the first passband starts at 0 (i.e. pass_zero is True)
            - fs/2 (the Nyquist frequency) if the first passband ends at fs/2
              (i.e the filter is a single band highpass filter); center of
              first passband otherwise

    Returns
    -------
        system :a tuple of array_like describing the system.
            The following gives the number of elements in the tuple and
            the interpretation:

                * (num, den)

    Raises
    ------
        ValueError
            -If any value in `Wn` is less than or equal to 0 or greater
             than or equal to 1, if the values in `Wn` are not strictly
             monotonically increasing, or if `n` is even but a passband
             includes the Nyquist frequency.
            -If the length of `Wn` equals to 1 but `ftype` is defined to
             other than 'default', 'low', 'high'.
            -If the length of `Wn` equals to 2 but `ftype` is defined to
             other than 'default', 'bandpass', 'stop'.
            -If the length of `Wn` more than 2 but `ftype` is defined to
             other than 'default', 'DC-0', 'DC-1'.
            -If `ftype` is other than 'default', 'low', 'bandpass', 'high',
             'stop', 'DC-0', 'DC-1'.

    BSD 3-Clause License

    Copyright (c) 2019, Yuki Fukuda
    All rights reserved.
    """

    # Default parameters
    filtertype = ['default', 'low', 'bandpass', 'high', 'stop', 'DC-0', 'DC-1']
    pass_zero = True

    # Filter type check
    if (ftype in filtertype) == False:
        raise ValueError("ftype must be 'default', 'low', 'bandpass', 'high'"
                         + ", 'stop', 'DC-0' or 'DC-1'.")

    # Filter length check
    if type(Wn) == float and (ftype in ['default', 'low', 'high']) == False:
        # When the length of Wn equals to 1.
        raise ValueError("If the length of Wn equals to 1, ftype must be"
                         + " 'default', 'low', or 'high'.")
    elif type(Wn) == list and len(Wn) == 2 and (ftype in ['default', 'bandpass', 'stop', 'DC-0', 'DC-1']) == False:
        # When the length of Wn equals to 2.
        raise ValueError("If the length of Wn equals to 2, ftype must be"
                         + " 'default', 'bandpass', 'stop', 'DC-0', 'DC-1'.")
    elif type(Wn) == list and len(Wn) >= 3 and (ftype in ['default', 'DC-0', 'DC-1']) == False:
        # When the length of Wn is greater than 2.
        raise ValueError("If the length of Wn is greater than 2, ftype must be"
                         + " 'default', 'DC-0', or 'DC-1'.")

    # Define default filter types
    if type(Wn) == float and ftype == 'default':
        # If the length of Wn equals to 1, the default filter type is low-pass
        ftype = 'low'

    if type(Wn) == list and len(Wn) == 2 and (ftype == 'default' or ftype == 'DC-0'):
        # If the length of Wn equals to 2, the default filter type is bandpass
        ftype = 'bandpass'

    if type(Wn) == list and len(Wn) >= 3 and ftype == 'default':
        # If the length of Wn is greater than 2, the default filter type is DC-0
        ftype = 'DC-0'

    if ftype in ['high', 'bandpass', 'DC-0']:
        pass_zero = False

    num = signal.firwin(n + 1, Wn, window=window, pass_zero=pass_zero,
                        scale=scaleopt)  # Numerator
    den = 1  # Denominator

    return num, den


def tukeywin(window_length, alpha=0.5):
    '''The Tukey window, also known as the tapered cosine window, can be regarded as a cosine lobe of width \alpha * N / 2
    that is convolved with a rectangle window of width (1 - \alpha / 2). At \alpha = 1 it becomes rectangular, and
    at \alpha = 0 it becomes a Hann window.

    We use the same reference as MATLAB to provide the same results in case users compare a MATLAB output to this function
    output

    Reference
    ---------
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html

    '''
    # Special cases
    if alpha <= 0:
        return np.ones(window_length)  # rectangular window
    elif alpha >= 1:
        return np.hanning(window_length)

    # Normal case
    x = np.linspace(0, 1, window_length)
    w = np.ones(x.shape)

    # first condition 0 <= x < alpha/2
    first_condition = x < alpha / 2
    w[first_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[first_condition] - alpha / 2)))

    # second condition already taken care of

    # third condition 1 - alpha / 2 <= x <= 1
    third_condition = x >= (1 - alpha / 2)
    w[third_condition] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[third_condition] - 1 + alpha / 2)))

    return w


def normalise(signal_f, amplitude):
    m = abs(signal_f).max(0)
    #     m = max(abs(signal_f))

    #     if m != 0:
    try:
        signal_final = signal_f * (np.sign(m) / m) * amplitude
    except:
        pass

    return signal_final


def generate_modulated_ersp(frequency, max_amplitude, modulation, modLatency, modWidth, modTaper,
                          modMinRelAmplitude):

    global samples, srate

    if len(frequency) == 1:
        print("Pure sine wave")  # not implemented.
    elif len(frequency) == 4:
        # design the filter
        n, Wn, beta, ftype = kaiserord(frequency, np.array([0, 1, 0]), np.array([0.05, 0.01, 0.05]), srate)
        num, den = fir1(n, Wn, ftype)

        # taking initial signal length of five times the filter order, but at
        # least as long as samples
        signal_f = np.random.rand(1, max([int(samples), 5 * len(num)])) - 0.5

        # filer the signal
        signal_filtered = signal.lfilter(num, den, signal_f)

        # removing padding (taking the middle segment)
        signal_filtered = signal_filtered[0, int(np.floor(signal_filtered.shape[1] / 2 - samples / 2) + 1):int(
            np.floor(signal_filtered.shape[1] / 2 + samples / 2) + 1)]

    # normalising such that max absolute amplitude = amplitude
    signal_filtered = normalise(signal_filtered, max_amplitude)

    # modulation section
    if modulation == 'burst' or modulation == 'invburst':
        latency = int(np.floor((modLatency / 1000) * srate) + 1)
        width = int(np.floor((modWidth / 1000) * srate))
        taper = modTaper

        # generating Tukey window for burst
        if width < 1:
            width = 0
        win = tukeywin(width, taper)
        win = np.transpose(win)

        # positioning window around latency
        if latency > np.ceil(width / 2):
            win = np.hstack((np.zeros(int(latency - np.ceil(width / 2)), dtype=np.float32), win))  # append horizontally
        else:
            win[1:int(np.ceil(width / 2) - latency + 1)] = 0

        # fitting Tukey window to signal
        if len(win) > samples:
            win = win[0:samples]
        elif len(win) < samples:
            win = np.append(win, np.zeros((1, int(samples) - len(win)), dtype=np.float32))

        # inverting in case of inverse burst
        if modulation == 'invburst':
            win = 1 - win

        if max(win) - min(win) != 0:
            # normalising between modMinRelAmplitude and 1
            win = modMinRelAmplitude + (1 - modMinRelAmplitude) * (win - min(win)) / (max(win) - min(win))

        elif np.all((win == 0)):
            # window is flat; if all-zero, it should be modMinRelAmplitude instead
            win = np.tile(modMinRelAmplitude, np.shape(win))

        # applying the modulation
        # print(signal_filtered.shape)
        # print(win.shape)
        signal_filtered = signal_filtered * win

    # make shape consistent with noise signal
    signal_filtered = np.expand_dims(signal_filtered, axis=1)

    return signal_filtered
# for testing
# x = GenerateERSPModulated([4.8, 5, 12, 12.2], 10, [], 'burst', 30, 20, 0.9, 0.25)
# plt.plot(x)


def generate_colored_noise(n, uniform_dist_range, noise_alpha):
    """
    Purpose: Generates a discrete colored noise vector of size n with power
    spectrum distribution of alpha
    White noise is sampled from Uniform (-range,range) distribution

    Usage: n - problem size
    uniform_dist_range - range of the underlying Uniform distribution
    noise_alpha - resulting colored noise has 1/f^alpha power spectrum
    """

    # Generate the coefficients Hk
    hfa = np.zeros((2 * n, 1), dtype=np.float32)
    hfa[0] = 1.0

    for i in range(1, n + 1):
        hfa[i] = hfa[i - 1] * (0.5 * noise_alpha + (i - 2 + 1)) / (
                    i - 1 + 1)  # coordinates are changed from Matlab code
    hfa[n + 1:2 * n + 1] = 0.0

    # Fill Wk with white noise
    wfa_1 = np.array(-uniform_dist_range + 2 * uniform_dist_range * np.random.rand(n, 1))
    wfa_2 = np.zeros((n, 1), dtype=np.float32)
    wfa = np.append(wfa_1, wfa_2, 0)  # append vertically

    # Perform the discrete Fourier transforms of Hk and Wk
    fh = np.fft.fft(hfa)
    fw = np.fft.fft(wfa)

    # Multiply the two complex vectors.
    fh = fh[1:n + 2]
    fw = fw[1:n + 2]
    fw = fh * fw

    # This scaling is introduced only to match the behavior of the Numerical Recipes code...
    fw[1] = fw[1] / 2
    fw[-1] = fw[-1] / 2

    # Take the inverse Fourier transform of the result.
    fw = np.append(fw, np.zeros((n - 1, 1), dtype=np.float32), 0)  # append vertically
    x = np.fft.ifft(fw)
    x = 2 * x[1:n + 1].real  # Discard the second half of IFT

    return x


def generate_noise_signal(color, amplitude):
    if color == 'white-unif':
        signal = np.random.rand(1, samples) - 0.5
    elif color == 'pink-unif':
        signal = generate_colored_noise(samples, 1, 1)
    elif color == 'brown-unif':
        signal = generate_colored_noise(samples, 1, 2)
    elif color == 'blue-unif':
        signal = generate_colored_noise(samples, 1, -1)
    elif color == 'purple-unif':
        signal = generate_colored_noise(samples, 1, -2)

    # center around zero
    signal = signal - np.mean(signal)

    # normalising to have the maximum (or minimum) value be (-)amplitude
    signal = normalise(signal, amplitude)

    return signal


def create_source_locations(location, absolute_mode):
    global pos
    selected_positions = []
    if absolute_mode == False:
        for i in location:
            distances = ((pos[:, 0] - i[0]) ** 2) + ((pos[:, 1] - i[1]) ** 2) + ((pos[:, 2] - i[2]) ** 2) ** 0.5
            selected_positions.append(np.argmin(distances))
    elif absolute_mode == True:
        pos_rounded = np.around(pos).astype(int).tolist()  # round to whole numbers
        for i in location:
            if i in pos_rounded:
                selected_positions.append(pos_rounded.index(i))
                break

    return selected_positions


def create_component(location, n, signal, component_list, absolute_mode, mode):
    """
    Compiles the required information to form a brain activity component.

    Input:
    location - desired location x, y, z coordinates of the activity, in a nested list format OR simply 'random'.
    n - relevant for when using 'random' for location; defines how many components to generate.
    signal - defines the signal to assign to the location(s).
    component_list - defines the component list to append the output to. Should be [] in first usage.
    leadfield - defines the leadfield - assign the output of loadmat to this.
    absolute_mode - set to True in order to disable the search nearest point function and instead strictly follow the
                    list of coordinates given in the 'location' argument when searching.

    Output:
    component_list - updated list of components
    """
    global pos, orientation
    # lf = leadfield['nyhead_lf_test'][0][0][0]  # extract the leadfield matrix
    # orientation = leadfield['nyhead_lf_test'][0][0][1]  # default orientations (orthogonal)
    # pos = leadfield['nyhead_lf_test'][0][0][2]
    # chanlocs = leadfield['nyhead_lf_test'][0][0][3]

    selected_positions = []

    if mode == 'sereega' or mode == 'SEREEGA':
        if location == 'random':
            selected_positions = random.sample(range(pos.shape[0] + 1), n)
        #  if a location is specified through coordinates or list of coordinates, search nearest points
        elif absolute_mode == False:
            for i in location:
                distances = ((pos[:, 0] - i[0]) ** 2) + ((pos[:, 1] - i[1]) ** 2) + ((pos[:, 2] - i[2]) ** 2) ** 0.5
                selected_positions.append(np.argmin(distances))
        elif absolute_mode == True:
            pos_rounded = np.around(pos).astype(int).tolist()  # round to whole numbers
            # for i in location:
            #     if i in pos_rounded:
            #         selected_positions.append(pos_rounded.index(i))
            selected_positions = [pos_rounded.index(i) for i in location if i in pos_rounded]

        for i in selected_positions:
            component = {
                "sourceIdx": i,
                "signal": signal,
                "projection": lf[:, i, :],
                "orientation": orientation[i],
                "position": pos[i],
            }
            component_list.append(component)

    elif mode == 'brainstorm' or mode == 'brainnetome' or mode == 'openmeeg':
        if location == 'random':
            selected_positions = random.sample(range(pos.shape[0] + 1), n)
        elif absolute_mode == False:
            print("absolute coordinate mode must be enabled for brainstorm/brainnetome method")
        elif absolute_mode == True:
            # assuming M1_left and M1_right inputs into location argument are vertex indices
            # then, selected_positions is already provided in location argument.
            selected_positions = location[:]

        for i in selected_positions:
            component = {
                "sourceIdx": i,
                "signal": signal,
                "projection": lf[:, i, :],
                "orientation": orientation[i],
                "position": pos[i],
            }
            component_list.append(component)
    return component_list


def add_sensornoise(data, mode, value):
    if mode == 'amplitude':
        data = data + normalise(rng.random(np.shape(data))*2-1, value)
    elif mode == 'snr':
        noise = normalise(rng.random(np.shape(data))*2-1, abs(data[:]).max(0))
        data = mix_data(data, noise, value)[0]
    return data


def mix_data(signal, noise, snr):
    """
    This function is used in the application of sensor-level signal-to-noise ratio.
    :param signal:
    :param noise:
    :param snr:
    :return:
    """
    originalsize = np.shape(signal)
    # if size(signal, 3) > 1
    #     signal = reshape(signal, size(signal, 1), []);
    #     noise = reshape(noise, size(noise, 1), []);
    # end
    try:
        if np.shape(signal)[2] > 1:  # if shape in 3rd dimension is larger than 1
            # porting tips: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
            #          and  http://mathesaurus.sourceforge.net/matlab-numpy.html
            print("3D signal detected")
            signal = signal.reshape(np.shape(signal)[0], [], order='F').copy()
            noise = noise.reshape(np.shape(signal)[0], [], order='F').copy()
    except IndexError:  # if original shape doesn't have a 3rd dimension
        pass

    meanamplitude = np.mean(abs(np.append(signal[:], noise[:], 0)))

    # scaling signal and noise, adding together
    signal = snr * signal / np.linalg.norm(signal, 'fro')
    noise = (1 - snr) * noise / np.linalg.norm(noise, 'fro')
    data = signal + noise

    # attempting to keep similar overall amplitude;
    # returning original signals at same scale
    scale = meanamplitude / np.mean(abs(data[:]))
    data = data * scale
    signal = signal * scale
    noise = noise * scale

    if np.size(originalsize) > 2:
        data = np.reshape(data, originalsize)
        signal = np.reshape(signal, originalsize)
        noise = np.reshape(noise, originalsize)

    # calculating dB value
    db = 10 * np.log10((np.sqrt(np.mean(signal[:])) / np.sqrt(np.mean(noise[:]))) ** 2)
    # print("SNR " + str(snr) + "= " + str(db) + " dB")

    return data, db, signal, noise

# def utl_add_variability()


def generate_scalp_data(component_list, sensorNoise):
    global srate, leadfield, chanlocs, lf

    scalpdata = np.zeros((len(chanlocs[0]), int(samples)), dtype=np.float32)
    componentdata = np.zeros((len(chanlocs[0]), int(samples), len(component_list)), dtype=np.float32)

    for i in range(len(component_list)):
        # obtaining component's signal
        componentsignal = component_list[i]["signal"]

        # obtaining the source
        sourceidx = component_list[i]["sourceIdx"]

        # obtaining orientation
        orient = component_list[i]["orientation"]

        # projecting signal
        componentdata[..., i] = project_activity(lf, componentsignal, sourceidx, orient, component_list, normaliseLeadfield=True,
                                                 normaliseOrientation=True)

    # Combining projected component signals into single epoch (sum along components)
    # scalpdata[:, :] = np.sum(componentdata, 2)  # sum along last axis
    scalpdata[:, :] = np.matmul(componentdata, np.ones((componentdata.shape[-1],)))  # fastest (hint from https://github.com/numpy/numpy/issues/16158 and comment from alex.jordan in https://math.stackexchange.com/questions/409460/multiplying-3d-matrix)
    # scalpdata[:, :] = contract("ijk->ij", componentdata)

    # Adding sensor noise
    # scalpdata = utl_add_sensornoise(scalpdata, 'amplitude', sensorNoise)
    # scalpdata = utl_add_sensornoise(scalpdata, 'snr', 2)

    # eeg = plt.plot(scalpdata)
    # plt.show()
    return scalpdata


def project_activity(lf, componentsignal, sourceidx, orient, component_list, normaliseLeadfield, normaliseOrientation):
    # Getting leadfield
    projection = np.squeeze(lf[:, sourceidx, :])
    # Alternative way that seems kinda slower for some reason (but actually makes use of the component_list):
    # projection = np.squeeze(next(item["projection"] for item in component_list if item["sourceIdx"] == sourceidx))

    # if normaliseLeadfield == True:
    #     #         projection = utl_normalise(projection, 1)  # to fix utl_normalise
    #     pass
    #
    # if normaliseOrientation == True:
    #     #         orientation = utl_normalise(orientation, 1)
    #     pass

    oriented_signal = (componentsignal * orient).T
    projdata = np.dot(projection, oriented_signal)  # faster than @ or matmul due to small n

    return projdata


def import_atlas(filename, mode):
    if mode == 'SEREEGA' or mode == 'sereega' or mode == 'xjview':
        brodmann = loadmat(filename)
        brodmann_4 = brodmann['wholeMaskMNIAll']['brodmann_area_4'][0][0]  # Primary motor cortex

        mask_for_left = brodmann_4[:, 0] < 0
        mask_for_right = brodmann_4[:, 0] > 0

        M1_left = brodmann_4[mask_for_left, :].tolist()
        M1_right = brodmann_4[mask_for_right, :].tolist()

        # convert to indices
        M1_left = create_source_locations(M1_left, True)
        M1_right = create_source_locations(M1_right, True)

    elif mode == 'openmeeg' or mode == 'brainstorm' or mode == 'brainnetome':
        area_def = loadmat(filename)
        M1_left = area_def['area_defs']['left'][0][0][0].tolist()
        M1_right = area_def['area_defs']['right'][0][0][0].tolist()

    # returns vertex indices that correspond to leadfield pos item
    return M1_left, M1_right
