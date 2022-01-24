from math import exp
import numpy as np
from matplotlib import pyplot as plt
# import time
import math
import matplotlib
from scipy.spatial.distance import cosine


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


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




def sigmoidfunction_plot_init(gain, visualization_mode):
    fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(8, 5))

    move_figure(fig, 25, 150)

    for a in ax.flatten():
        a.set_ylim(-0.2, 1.2)
        a.set_xlim(-1, 1)
        a.grid()

    x = np.linspace(-1, 1, 100)
    y1 = []; y2 = []; y3 = []; y4 = []
    for i in x:
        y1.append(1 / (1 + exp(gain * (i-0.6))))
        y3.append(1 / (1 + exp(-gain * (i+0.6))))
    y2 = y1[:]
    y4 = y1[:]

    ax[0, 0].title.set_text("C3 (horizontal)")
    ax[1, 0].title.set_text("C3 (vertical)")
    ax[0, 1].title.set_text("C4 (horizontal)")
    ax[1, 1].title.set_text("C4 (vertical)")

    for a in ax.flatten():
        a.set_xlabel("unit velocity")
        a.set_ylabel("Amod")
        a.set_xlim((-1, 1))
        a.set_ylim((-0.2, 1.2))

    line1, = ax[0, 0].plot(x, y1)
    line2, = ax[1, 0].plot(x, y2)
    line3, = ax[0, 1].plot(x, y3)
    line4, = ax[1, 1].plot(x, y4)
    fig.tight_layout()
    fig.canvas.draw()

    p1, = ax[0, 0].plot(0, 0, 'go'); arrow1 = ax[0, 0].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
    p2, = ax[1, 0].plot(0, 0, 'go'); arrow2 = ax[1, 0].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
    p3, = ax[0, 1].plot(0, 0, 'go'); arrow3 = ax[0, 1].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
    p4, = ax[1, 1].plot(0, 0, 'go'); arrow4 = ax[1, 1].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)

    p = [p1, p2, p3, p4]; arrow = [arrow1, arrow2, arrow3, arrow4]
    ax[1, 0].legend(handles=[arrow4], labels=["Optimal direction"], loc='lower right',
                 bbox_to_anchor=(1.47, -0.15), fancybox=False, shadow=False, ncol=1)

    background = [fig.canvas.copy_from_bbox(a.bbox) for a in ax.flatten()]

    if visualization_mode:
        plt.show(block=False)

    return fig, ax, background, p, arrow


def velocity_to_mod_amplitude(vx, vy, mode, gain, visualization_mode, **kwargs):
    """
    New approach which maps incoming (scaled) velocity into modminRelAmp (Amod parameter) for ERSP generation.
    Amin is the lower limit for minModRelAmplitude, dictates source-level signal-to-noise ratio.
    """

    fig = kwargs.get('fig', None)
    ax = kwargs.get('ax', None)
    background = kwargs.get('background', None)
    p = kwargs.get('p', None)
    arrow = kwargs.get('arrow', None)
    latest_optimal = kwargs.get('latest_optimal', None)

    # Approach 1: (original)
    i = 0
    while abs(vx) > 1 or abs(vy) > 1:
        vx = vx / 2
        vy = vy / 2
        i = i + 1
    scaling_factor = i  # scaling_factor to be multiplied back to the output velocity after decoder

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
        gain = [2, 1, -0.5, 10]
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

    if visualization_mode:
        # animation
        for bg in background:
            fig.canvas.restore_region(bg)
        p[0].set_data(vx, C3_x)
        p[1].set_data(vy, C3_y)
        p[2].set_data(vx, C4_x)
        p[3].set_data(vy, C4_y)

        arrow[0] = ax[0, 0].arrow(0, 0, 0.5*np.sign(latest_optimal[0]), 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
        arrow[1] = ax[1, 0].arrow(0, 0, 0.5*np.sign(latest_optimal[1]), 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
        arrow[2] = ax[0, 1].arrow(0, 0, 0.5*np.sign(latest_optimal[0]), 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
        arrow[3] = ax[1, 1].arrow(0, 0, 0.5*np.sign(latest_optimal[1]), 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)

        ax[0, 0].draw_artist(p[0])
        ax[0, 0].draw_artist(arrow[0])
        ax[1, 0].draw_artist(p[1])
        ax[1, 0].draw_artist(arrow[1])
        ax[0, 1].draw_artist(p[2])
        ax[0, 1].draw_artist(arrow[2])
        ax[1, 1].draw_artist(p[3])
        ax[1, 1].draw_artist(arrow[3])

        # fig.canvas.update()
        # fig.canvas.flush_events()
        for a in ax.flatten():
            fig.canvas.blit(a.bbox)

        # somehow removing doesn't work, so workaround
        p[0].set_data(50, 50); arrow[0] = ax[0, 0].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
        p[1].set_data(50, 50); arrow[1] = ax[1, 0].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
        p[2].set_data(50, 50); arrow[2] = ax[0, 1].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)
        p[3].set_data(50, 50); arrow[3] = ax[1, 1].arrow(0, 0, 0, 0, width=0.01, length_includes_head=True, head_width=0.05, head_length=0.05, color='r', zorder=1000)

    return amp_C3, amp_C4, scaling_factor


def decodedvelocity_plot_init(visualization_mode):
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(5, 5))
    # fig.canvas.manager.window.setGeometry(1000, 300, 500, 500)  # causes problems with the circle for some reason
    move_figure(fig, 1350, 400)

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    # ax.axis('equal')
    # Unit circle (if needed)
    # circle = plt.Circle((0, 0), 1.0, color='black', fill=False)  # matplotlib.patches.Circle

    ax.title.set_text("Velocity Vector Visualization")
    ax.set_xlabel("x position"); ax.set_ylabel("y position")
    fig.tight_layout()

    # ax.add_artist(circle)
    fig.canvas.draw()

    # Initialize arrows
    true_v = plt.arrow(0, 0, 0, 0, length_includes_head=True, head_width=2, head_length=4, color='g')
    decoded_v = plt.arrow(0, 0, 0, 0, length_includes_head=True, head_width=2, head_length=4, color='b')  # initialize vectors at (0, 0).
    optimal_v = plt.arrow(0, 0, 0, 0, length_includes_head=True, head_width=2, head_length=4, color='r')  # initialize vectors at (0, 0).
    v = [true_v, decoded_v, optimal_v]

    leg = ax.legend(v, ["True", "Decoded", "Optimal"])
    background = fig.canvas.copy_from_bbox(ax.bbox)

    if visualization_mode:
        plt.show(block=False)

    return fig, ax, background, v, leg


def decodedvelocity_plot_update(true_vx, true_vy, decoded_vx, decoded_vy, optimal_vx, optimal_vy, fig, ax, background, v, leg):
    # while abs(true_vx) > 1 or abs(true_vy) > 1:
    #     true_vx = true_vx / 2
    #     true_vy = true_vy / 2
    #
    #     optimal_vx = optimal_vx / 2
    #     optimal_vy = optimal_vy / 2

    true_length = (true_vx ** 2 + true_vy ** 2) ** 0.5
    if true_length == 0:
        true_length = 0.1
    decoded_length = (decoded_vx ** 2 + decoded_vy ** 2) ** 0.5
    if decoded_length == 0:
        decoded_length = 0.1     # dividing by length of a vector then multiplying by desired length (1 in this case) forces the vector to have that length

    fig.canvas.restore_region(background)
    v[0] = plt.arrow(0, 0, true_vx, true_vy, length_includes_head=True, head_width=2, head_length=4, color='g')
    v[1] = plt.arrow(0, 0, decoded_vx, decoded_vy, length_includes_head=True, head_width=2, head_length=4, color='b')
    v[2] = plt.arrow(0, 0, optimal_vx, optimal_vy, length_includes_head=True, head_width=2, head_length=4, color='r')

    ax.draw_artist(v[0])
    ax.draw_artist(v[1])
    ax.draw_artist(v[2])
    ax.draw_artist(leg)

    # fig.canvas.update()
    # fig.canvas.flush_events()
    fig.canvas.blit(ax.bbox)

    # somehow removing doesn't work, so workaround
    v[0] = plt.arrow(0, 0, 0, 0, length_includes_head=True, head_width=2, head_length=4, color='g')
    v[1] = plt.arrow(0, 0, 0, 0, length_includes_head=True, head_width=2, head_length=4, color='b')
    v[2] = plt.arrow(0, 0, 0, 0, length_includes_head=True, head_width=2, head_length=4, color='r')

    return
