from math import exp
from numpy.random import normal


def sigmoidfunction_alt(v, mode, gain):
    if mode == 'C3_x':
        a = 1 / (1 + exp(gain * v))
    elif mode == 'C4_x':
        a = 1 / (1 + exp(-gain * v))
    elif mode == 'C3_y' or mode == 'C4_y':
        a = 1 / (1 + exp(gain * v))
    return a


def sigmoidfunction_classic(v, mode, gain):
    if mode == 'C3_x' or mode == 'C3_y' or mode == 'C4_y':
        a = 1 / (1 + exp(gain * (v-0.6)))  # v-0.6
    elif mode == 'C4_x':
        a = 1 / (1 + exp(-gain * (v+0.6)))  # v+0.6
    return a


def sigmoidfunction_perturbed(v, mode, gain):
    if mode == 'C3_x':
        a = 1 / (1 + exp(gain[0] * v))
    elif mode == 'C3_y':
        a = 1 / (1 + exp(gain[1] * v))
    elif mode == 'C4_x':
        a = 1 / (1 + exp(gain[2] * v))
    elif mode == 'C4_y':
        a = 1 / (1 + exp(gain[3] * v))
    return a


def velocity_to_mod_amplitude(vx, vy, mode, gain):
    """
    New approach which maps incoming (scaled) velocity into modminRelAmp (Amod parameter) for ERSP generation.
    Amin is the lower limit for minModRelAmplitude, dictates source-level signal-to-noise ratio.
    """

    # Approach 1: (original)
    i = 0
    while abs(vx) > 1 or abs(vy) > 1:
        vx = vx / 2
        vy = vy / 2
        i = i + 1
    scaling_factor = i  # scaling_factor to be multiplied back to the output velocity after decoder
    # print(vx)

    # Apply random noise!
    vx = vx + normal(0, 0.325, 1)
    vy = vy + normal(0, 0.325, 1)

    # Approach 2: Set maximum velocity as 100
    # scaling_factor = 375
    # print(vx, vy)
    # if abs(vx) >= scaling_factor:
    #     vx = np.sign(vx) * scaling_factor
    # if abs(vy) >= scaling_factor:
    #     vy = np.sign(vy) * scaling_factor
    # vx = vx / scaling_factor
    # vy = vy / scaling_factor
    # print(vx, vy)

    # working gain was 7
    if mode == 'classic':
        C3_x = sigmoidfunction_classic(vx, 'C3_x', gain); C3_y = sigmoidfunction_classic(vy, 'C3_y', gain)
        C4_x = sigmoidfunction_classic(vx, 'C4_x', gain); C4_y = sigmoidfunction_classic(vy, 'C4_y', gain)
    elif mode == 'differential' or mode == 'diff':
        C3_x = sigmoidfunction_alt(vx, 'C3_x', gain); C3_y = sigmoidfunction_alt(vy, 'C3_y', gain)
        C4_x = sigmoidfunction_alt(vx, 'C4_x', gain); C4_y = sigmoidfunction_alt(vy, 'C4_y', gain)
    elif mode == 'perturbed':
        gain = [2, 7, -3, 7]
        C3_x = sigmoidfunction_perturbed(vx, 'C3_x', gain)
        C3_y = sigmoidfunction_perturbed(vy, 'C3_y', gain)
        C4_x = sigmoidfunction_perturbed(vx, 'C4_x', gain)
        C4_y = sigmoidfunction_perturbed(vy, 'C4_y', gain)

    # print(gain, C3_x, C3_y, C4_x, C4_y)

    # Approach 1:
    amp_C3 = (C3_x + C3_y) / 2
    amp_C4 = (C4_x + C4_y) / 2

    # Approach 2:
    # Return C3_x, C3_y, C4_x, C4_y and individually assign source

    return amp_C3, amp_C4, scaling_factor

