import pygame, pygame.gfxdraw, pygame.font, random, math, os
from datetime import datetime
import numpy as np
import pandas as pd
import time
from itertools import compress
from tkinter import simpledialog, filedialog
from pygame.locals import *
from matplotlib import pyplot as plt

from ui import GetParams
from eeg import VelocityMapping, SourceSignals
from task import Targets
from encryption.encrypt_data import encrypt_data
from decode.decoder_arpsd import decoder_arpsd
from decode.spatial_filter import spatial_filter

params = GetParams.user_input()
# params = GetParams.dev_input()  # uncomment to skip user-interface

max_target_velocity = params['max_target_velocity']
max_cursor_velocity = params['max_cursor_velocity']
trialLength = params['trialLength']
trialCount = params['trialCount']
visualization_mode = params['visualization_mode']
summary_visualization_mode = params['summary_visualization_mode']
normalize_mode = params['normalize_mode']
velocity_gain = params['velocity_gain']
game_mode = params['game_mode']
leadfield_mode = params['leadfield_mode']
experiment_mode = params['experiment_mode']
target_physics_mode = params['target_physics_mode']

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (515, 100)
pygame.init()
test_mode = False
frames = 0
wait_frames = 0
myfont = pygame.font.SysFont('Calibri', 20)
textsurface = myfont.render('Waiting for next trial...', False, (0, 0, 0))
textsurface_calibration = myfont.render('Waiting for calibration trial...', False, (0, 0, 0))
textsurface_calibration_new = myfont.render('Calibration trial ready.', False, (0, 0, 0))
textsurface_new = myfont.render('Next trial ready.', False, (0, 0, 0))
textsurface2_new = myfont.render('Please return mouse to center of desk and press the spacebar when ready to start.', False, (0, 0, 0))
score = 0

# Set-up the screen
screen_width = Targets.screen_width
screen_height = Targets.screen_height

flags = FULLSCREEN | HWSURFACE | HWACCEL | DOUBLEBUF
import ctypes
ctypes.windll.user32.SetProcessDPIAware()
# true_res = (ctypes.windll.user32.GetSystemMetrics(0),ctypes.windll.user32.GetSystemMetrics(1))
screen = pygame.display.set_mode((screen_width, screen_height))
# screen = pygame.display.set_mode((screen_width, screen_height), flags)
screen.set_alpha(None)
target_loc = "left"
target_width = 25
target_height = 700

# Set-up the sprites
playerSprite = pygame.sprite.Group()
targetSprite = pygame.sprite.Group()  # list of target sprites
chosenTarget = pygame.sprite.GroupSingle()
playerStepCapturerSprite = pygame.sprite.GroupSingle()
playerStepModulatorSprite = pygame.sprite.GroupSingle()

# Initialize control variables
x_control_buffer = []; y_control_buffer = []
# vis_xcontrol_b4_dwell = []; vis_ycontrol_b4_dwell = []
# vis_xcontrol_dwell = []; vis_ycontrol_dwell = []
# vis_vnorms_b4_cap = []; vis_vnorms_capped = []
# vis_dwell_class_over_time_x = []; vis_dwell_class_over_time_y = []
# vis_optimal_over_time = []
tvcv_order = []
reps_per_velocity_limit_ratio = 3

# Initialize container for saved trajectories
if game_mode == "CP":
    targetCount = 1
elif game_mode == "DT_rect":
    targetCount = 11

playerPath = [[[] for targets in range(targetCount * 4)] for trials in range(trialCount)]
playerVel = [[[] for targets in range(targetCount * 4)] for trials in range(trialCount)]
targetPath = [[[] for targets in range(targetCount * 4)] for trials in range(trialCount)]
targetVel = [[[] for targets in range(targetCount * 4)] for trials in range(trialCount)]
trueVel = [[[] for targets in range(targetCount * 4)] for trials in range(trialCount)]
targetNumber = [[[] for targets in range(targetCount * 4)] for trials in range(trialCount)]

# vis_true_x_over_time = []; vis_true_y_over_time = []

# Atlas and leadfield configuration
SourceSignals.import_lf_within_source_signals(leadfield_mode)

# Import atlas
if leadfield_mode == 'sereega':
    M1_left, M1_right = SourceSignals.import_atlas('data/atlas/brodmann_area_def.mat', 'SEREEGA')
elif leadfield_mode == 'brainstorm':
    M1_left, M1_right = SourceSignals.import_atlas('data/atlas/brainnetome_area_def.mat', 'brainstorm')

# Epoch configuration
srate, epochLength, samples = SourceSignals.epoch_config(srate=512, epochLength=200)

# Import leadfield
if leadfield_mode == 'sereega':
    leadfield, lf, orientation, pos, chanlocs = SourceSignals.import_leadfield('data/leadfield/32_channel_nyhead_lf.mat', 'sereega')
elif leadfield_mode == 'brainstorm':
    leadfield, lf, orientation, pos, chanlocs = SourceSignals.import_leadfield('data/leadfield/openmeeg_fsaverage_lf.mat', 'brainstorm')

# Import channel labels and choose channels for spatial Laplacian filter
channel_labels = []
for i in chanlocs[0]:
    channel_labels.append(i[1][0])
C3_relevant_indexes = [channel_labels.index(c) for c in ['FC5', 'FC1', 'CP5', 'CP1'] if c in channel_labels]
C4_relevant_indexes = [channel_labels.index(c) for c in ['FC2', 'FC6', 'CP2', 'CP6'] if c in channel_labels]
C3_index = [channel_labels.index(c) for c in ['C3'] if c in channel_labels]
C4_index = [channel_labels.index(c) for c in ['C4'] if c in channel_labels]
channelSelection = np.array([[0] * len(channel_labels), [0] * len(channel_labels)], dtype=np.float32)
for i in C3_relevant_indexes:
    channelSelection[0][i] = -0.25
for i in C4_relevant_indexes:
    channelSelection[1][i] = -0.25
for i in C3_index:
    channelSelection[0][i] = 1
for i in C4_index:
    channelSelection[1][i] = 1

# Prepare background signal
noise = SourceSignals.generate_noise_signal('brown-unif', 5.0)

# Prepare background/resting components
if leadfield_mode == 'sereega':
    component_list_background = SourceSignals.create_component('random', 100, noise, component_list=[], absolute_mode=False, mode='sereega')
elif leadfield_mode == 'brainstorm':
    component_list_background = SourceSignals.create_component('random', 100, noise, component_list=[], absolute_mode=False, mode='brainstorm')

# Sigmoid function parameter
Amin = 0.01

pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

# Global experiment progress tracking variables.
trial_no = 0
target_no = 0

class PlayerStepCapturer(pygame.sprite.Sprite):
    """
    This is the hidden player character that exists to track relative velocity movement.
    """
    def __init__(self):
        global screen_width, screen_height
        pygame.sprite.Sprite.__init__(self)
        self.radius = 25

        self.image = pygame.Surface((self.radius * 2, self.radius * 2))  # need a better way of drawing the circle
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))

        self.rect = self.image.get_rect()

        self.rect.x = screen_width // 2 - self.radius
        self.rect.y = screen_height // 2 - self.radius

        self.pos_t1 = (screen_width // 2 - self.radius, screen_height // 2 - self.radius)
        self.pos_t0 = (screen_width // 2 - self.radius, screen_height // 2 - self.radius)

    def update(self, frames, trial_no, target_no, trueVel):
        self.pos_t1 = pygame.mouse.get_pos()

        step_velocity_vector = (self.pos_t1[0] - self.pos_t0[0], self.pos_t1[1] - self.pos_t0[1])

        self.rect.x = self.pos_t1[0]
        self.rect.y = self.pos_t1[1]

        # Reset back to midpoint
        # pygame.mouse.set_pos((screen_width // 2 - self.radius, screen_height // 2 - self.radius))
        self.pos_t1 = (screen_width // 2 - self.radius, screen_height // 2 - self.radius)
        pygame.mouse.set_pos(self.pos_t1)

        trueVel[trial_no][target_no].append([step_velocity_vector[0], step_velocity_vector[1]])

        return step_velocity_vector, trueVel  # This velocity will be passed on to gain modulation


class PlayerStepModulator(pygame.sprite.Sprite):
    """
        This character will move with a velocity vector measured by the PlayerStepCapturer, but shortened.
    """
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        global velocity_gain
        self.radius = 25

        self.velocity_gain = velocity_gain

        self.image = pygame.Surface((self.radius * 2, self.radius * 2))  # need a better way of drawing the circle
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))

        # pygame.draw.circle(self.image, (0, 0, 0), [self.radius, self.radius], self.radius)
        self.rect = self.image.get_rect()

        self.rect.x = screen_width // 2 - self.radius
        self.rect.y = screen_height // 2 - self.radius

        # build position tracker since rect does not support float mathematics.
        self.current_pos = [float(self.rect.x), float(self.rect.y)]

    def update(self, v1):
        modulated_v = (v1[0] * self.velocity_gain, v1[1] * self.velocity_gain)
        # modulated_v will be the input to the decoder.

        self.current_pos += modulated_v

        self.rect.x = int(self.current_pos[0])
        self.rect.y = int(self.current_pos[1])

        return modulated_v, self.velocity_gain


class Player(pygame.sprite.Sprite):
    """
    This is the player character that shows the decoded position.
    """
    def __init__(self, currentTrial):
        global playerPath, trialLength, Amin, max_cursor_velocity, visualization_mode
        pygame.sprite.Sprite.__init__(self)
        self.radius = 20
        self.edge_threshold = 0.01
        self.image = pygame.Surface((self.radius * 2, self.radius * 2))  # need a better way of drawing the circle
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))

        pygame.draw.circle(self.image, (0, 0, 255), [self.radius, self.radius], self.radius)
        self.rect = self.image.get_rect()

        self.rect.x = screen_width // 2 - self.radius
        self.rect.y = screen_height // 2 - self.radius

        # build position tracker since rect does not support float mathematics.
        self.current_pos = [float(self.rect.x), float(self.rect.y)]

        self.m = 280
        self.v = pygame.Vector2()
        self.v.xy = 0.0001, 0.0001  # Avoid division by zero

        self.f_external = pygame.Vector2()
        # self.f_external.xy = np.random.normal(0, 100), np.random.normal(0, 100)

        self.friction_coeff = 20
        self.f_friction = pygame.Vector2()
        self.drag_coeff = 0.4
        self.f_drag = pygame.Vector2()

        self.v = pygame.Vector2()
        self.v.xy = 0.0001, 0.0001  # Avoid division by zero

        # Plotting function initialization
        # get gain from Amin by setting v = 1 and setting gain as the subject of equation
        # To-do: Check if sigmoid shape is fine in Amin parameter choice edge cases
        # self.gain = (1 / (1 - 0.6)) * np.log((1 / Amin) - 1)  # for C4_x, minus sign is applied within sigmoid function
        self.gain = 5
        print("Calculated sigmoid function gain: " + str(self.gain))

        if currentTrial > 0 and visualization_mode:
        # if currentTrial >= 0:
            self.fig, self.ax, self.background, self.p, self.arrow = VelocityMapping.sigmoidfunction_plot_init(self.gain, visualization_mode)
            self.figv, self.axv, self.backgroundv, self.vector_arrow_object, self.leg = VelocityMapping.decodedvelocity_plot_init(visualization_mode)

        self.vx_input = []; self.vy_input = []
        self.angle_decoded_vs_true = []; self.angle_decoded_vs_optimal = []
        self.error_decoded_vs_true = []; self.error_decoded_vs_optimal = []
        self.latest_optimal = [0, 0]

        self.max_cursor_velocity = max_cursor_velocity

    def update(self, v, modulation_gain, frames, trial_no, target_no, playerPath, playerVel):
        global d, x_control_buffer, y_control_buffer, \
            component_list_background, visualization_mode, vis_xcontrol_b4_dwell, vis_ycontrol_b4_dwell, \
            vis_xcontrol_dwell, vis_ycontrol_dwell, vis_vnorms_b4_cap, vis_vnorms_capped, \
            vis_true_x_over_time, vis_true_y_over_time

        if test_mode == True:
            v = pygame.mouse.get_rel()  # if just testing, just use mouse movement
        else:
            # Obtain C3 and C4 amplitudes from recorded velocity
            try:
                self.vx_input.append(v[0]); self.vy_input.append(-v[1])
                amp_C3, amp_C4, scaling_factor = VelocityMapping.velocity_to_mod_amplitude(v[0], v[1], 'diff', self.gain, visualization_mode, fig=self.fig, ax=self.ax, background=self.background, p=self.p, arrow=self.arrow, latest_optimal=self.latest_optimal)
            except AttributeError:  # if calibration trial:
                amp_C3, amp_C4, scaling_factor = VelocityMapping.velocity_to_mod_amplitude(v[0], v[1], 'diff', self.gain, visualization_mode)
            except TypeError:  # catches the case where velocity is 'NoneType'
                v = (0.0, 0.0)
                amp_C3, amp_C4, scaling_factor = VelocityMapping.velocity_to_mod_amplitude(v[0], v[1], 'diff', self.gain, visualization_mode, fig=self.fig, ax=self.ax, background=self.background, p=self.p, arrow=self.arrow, latest_optimal=self.latest_optimal)

            # Prepare signals using these amplitudes
            # source location assignment has been moved outside the update loop
            freqRange = [3, 5, 12, 14]
            target_C3_x = SourceSignals.generate_modulated_ersp(freqRange, 5, 'invburst',
                                                             5, 20, 0.5, amp_C3)
            target_C3_y = SourceSignals.generate_modulated_ersp(freqRange, 5, 'invburst',
                                                             5, 20, 0.5, amp_C3)
            target_C4_x = SourceSignals.generate_modulated_ersp(freqRange, 5, 'invburst',
                                                             5, 20, 0.5, amp_C4)
            target_C4_y = SourceSignals.generate_modulated_ersp(freqRange, 5, 'invburst',
                                                               5, 20, 0.5, amp_C4)

            # Create components
            active_component_list = component_list_background[:]  # copy over background components, to which active components are appended.

            for i in M1_left:
                component = {
                    "sourceIdx": i,
                    "signal": target_C3_x,
                    "projection": lf[:, i, :],
                    "orientation": orientation[i],
                    "position": pos[i],
                }
                active_component_list.append(component)
                component = {
                    "sourceIdx": i,
                    "signal": target_C3_y,
                    "projection": lf[:, i, :],
                    "orientation": orientation[i],
                    "position": pos[i],
                }
                active_component_list.append(component)

            for i in M1_right:
                component = {
                    "sourceIdx": i,
                    "signal": target_C4_x,
                    "projection": lf[:, i, :],
                    "orientation": orientation[i],
                    "position": pos[i],
                }
                active_component_list.append(component)
                component = {
                    "sourceIdx": i,
                    "signal": target_C4_y,
                    "projection": lf[:, i, :],
                    "orientation": orientation[i],
                    "position": pos[i],
                }
                active_component_list.append(component)

            # Generate scalp data
            eeg = SourceSignals.generate_scalp_data(active_component_list, sensorNoise=0.0)

            # Spatial filtering and decoding
            eeg_spatial_filtered = spatial_filter(eeg, channelSelection)
            x_control, y_control, x_control_buffer, y_control_buffer, d = decoder_arpsd(eeg_spatial_filtered, x_control_buffer, y_control_buffer, normalize_mode=normalize_mode)

            # Restore scaling to the control vectors
            # this scaling number can shift the mean of the velocity distribution
            x_control = x_control * scaling_factor* 3
            y_control = y_control * scaling_factor* 3

            if self.v.length() == 0:
                self.v.xy = 0.000001, 0.000001

            if x_control == 0:
                x_control += 0.000001
            if y_control == 0:
                y_control += 0.000001

            # Force-based approach
            self.f_external.xy = x_control * self.m, y_control * self.m
            self.f_friction = -self.friction_coeff * (self.v / self.v.length())
            self.f_drag = -self.drag_coeff * self.v * self.v.length()
            self.v = self.v + self.f_drag / self.m + self.f_friction / self.m + self.f_external / self.m

            # Velocity-based approach - uncomment this single line (overwrites the force-based approach above)
            self.v.xy = x_control, y_control

            # Pass the velocity into edge collision check.
            if self.rect.x + self.v.x <= (screen_width - self.radius * 2) * self.edge_threshold:
                print("hit left wall")
                self.v.x = 0
            elif self.rect.x + self.v.x >= (screen_width - self.radius * 2) * (1 - self.edge_threshold):
                print("hit right wall")
                self.v.x = 0
            if self.rect.y + self.v.y <= (screen_height - self.radius * 2) * self.edge_threshold:
                print("hit top wall")
                self.v.y = 0
            elif self.rect.y + self.v.y >= (screen_height - self.radius * 2) * (1 - self.edge_threshold):
                print("hit bottom wall")
                self.v.y = 0

            # Threshold velocity with Cv
            v = np.array([[self.v.x], [self.v.y]], dtype=np.float32)
            # print("length of v before capping: " + str(np.linalg.norm(v)))
            if np.linalg.norm(v) > self.max_cursor_velocity:
                # print("max cursor velocity hit: " + str(self.max_cursor_velocity))
                control_max = np.array([[self.max_cursor_velocity], [self.max_cursor_velocity]], dtype=np.float32)
                new_v = control_max * v.transpose() / np.linalg.norm(v)
                self.v.x = new_v[0][0]
                self.v.y = new_v[0][1]

            # Finally, we have to update the position as well:
            self.current_pos[0] += self.v.x
            self.current_pos[1] += self.v.y

            self.rect.x = int(self.current_pos[0])
            self.rect.y = int(self.current_pos[1])

            playerPath[trial_no][target_no].append([self.rect.x, self.rect.y])
            playerVel[trial_no][target_no].append([self.v.x, self.v.y])

        return playerPath, playerVel

    def plot_error_vs_angle(self, bins=12, elements=['decoded_vs_true', 'decoded_vs_optimal', 'decoded_vs_true_counts', 'decoded_vs_optimal_counts']):
        # https://stackoverflow.com/questions/21619347/creating-a-python-histogram-without-pylab
        # create figure and axis objects with subplots()
        fig, ax = plt.subplots()

        if 'decoded_vs_true_counts' in elements:
            # Takes the angle list and bins it, with average error values attached to it
            bin_array = np.linspace(-np.pi, np.pi, bins)

            hist_avg_error = []
            hist_counts = []
            angles = np.array(self.angle_decoded_vs_true, dtype=np.float32)
            errors = self.error_decoded_vs_true[:]
            for i in range(len(bin_array) - 1):  # for each bin
                mask = (angles >= bin_array[i]) & (angles < bin_array[i + 1])
                indices = list(compress(range(len(angles[mask])), angles[mask]))
                sum_of_errors_in_bin = 0
                for idx in indices:
                    sum_of_errors_in_bin += errors[idx]
                avg_error_in_bin = sum_of_errors_in_bin / len(angles[mask])
                hist_avg_error.append(avg_error_in_bin)
                hist_counts.append(len(angles[mask]))

            ax.set_xlabel("Angle between decoded and true velocity vectors")
            ax.bar((bin_array[1:] + bin_array[:-1]) / 2., hist_counts, zorder=-100)
            ax.set_ylabel("Frequency", color='b')

        if 'decoded_vs_true' in elements:
            # Takes the angle list and bins it, with average error values attached to it
            bin_array = np.linspace(-np.pi, np.pi, bins)

            hist_avg_error = []; hist_counts = []
            angles = np.array(self.angle_decoded_vs_true, dtype=np.float32)
            errors = self.error_decoded_vs_true[:]
            for i in range(len(bin_array) - 1):  # for each bin
                mask = (angles >= bin_array[i]) & (angles < bin_array[i + 1])
                indices = list(compress(range(len(angles[mask])), angles[mask]))
                sum_of_errors_in_bin = 0
                for idx in indices:
                    sum_of_errors_in_bin += errors[idx]
                avg_error_in_bin = sum_of_errors_in_bin / len(angles[mask])
                hist_avg_error.append(avg_error_in_bin)
                hist_counts.append(len(angles[mask]))  # might be useful for seeing distribution of angles

            # twin object for two different y-axis on the sample plot
            ax2 = ax.twinx()
            ax2.plot((bin_array[1:] + bin_array[:-1]) / 2., hist_avg_error, 'r-', zorder=100)
            ax2.set_ylabel("Average error in vector length", color='r')

        plt.show()


def main():
    global score, target_loc, target_width, target_height, frames, summary_visualization_mode, wait_frames, x_control_buffer, \
        y_control_buffer, d, trial_no, target_no, playerPath, playerVel, targetPath, targetVel, trueVel

    pygame.display.set_caption("BCI Simulator")
    last_chosen_LR = "placeholder"
    last_chosen_DU = "placeholder"

    # configure the background of playing area
    background = pygame.Surface(screen.get_size())
    background.fill((255, 255, 255))
    screen.blit(background, (0, 0))

    pygame.mouse.set_visible(False)  # hide mouse

    datetime_of_trials = []

    print(game_mode)

    if experiment_mode:  # Experimental protocol goes here
        target_velocity_search_list = [13, 17, 21]
        cursor_velocity_search_list = [11, 17, 23]
        velocity_grid = []
        for Tv in target_velocity_search_list:
            for Cv in cursor_velocity_search_list:
                for n in range(reps_per_velocity_limit_ratio):
                    velocity_grid.append([Tv, Cv])
        print("Will be following from the following Tv/Cv ratios randomly: " + str(velocity_grid))

    currentTrial = 0
    if game_mode == "DT_rect":
        print("DT_rect selected, skipping circular reach-out calibration")
        currentTrial += 1

    clock = pygame.time.Clock()

    keepGoing = True
    start = False

    seconds_out_tracker = False
    screen.blit(textsurface_calibration, (200, 400))

    seconds_out = pygame.USEREVENT + 1  # creates "break time over" event

    for n in range(targetCount):
        block = Targets.BarTargetLeft()
        targetSprite.add(block)
        block = Targets.BarTargetRight()
        targetSprite.add(block)
        block = Targets.BarTargetUp()
        targetSprite.add(block)
        block = Targets.BarTargetDown()
        targetSprite.add(block)

    # main loop
    while keepGoing:
        clock.tick(30)

        if not start:
            chosenTarget.clear(screen, background)
            playerSprite.clear(screen, background)
            seconds_out_tracker = False
            if currentTrial > 0:
                screen.blit(textsurface, (200, 400))

            wait_frames += 1  # countdown to next trial

            if wait_frames > 100:
                screen.fill(pygame.Color("white"))
                if currentTrial == 0:
                    screen.blit(textsurface_calibration_new, (275, 400))
                elif currentTrial > 0:
                    screen.blit(textsurface_new, (282, 400))
                screen.blit(textsurface2_new, (20, 425))

                my_event = pygame.event.Event(seconds_out, message="Break time over")
                pygame.event.post(my_event)
                seconds_out_tracker = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    # Force quit by pressing 'p'
                    keepGoing = False

                if event.key == pygame.K_SPACE and not start and seconds_out_tracker:
                    starto = time.time()
                    now = datetime.now()
                    now = now.strftime("%Y-%m-%d-%H-%M-%S")
                    datetime_of_trials.append(now)

                    screen.fill(pygame.Color("white"))
                    playerSprite.clear(screen, background)
                    chosenTarget.clear(screen, background)
                    playerStepCapturerSprite.clear(screen, background)  # ?
                    playerStepModulatorSprite.clear(screen, background)  # ?

                    # Reset after an iteration
                    chosenTarget.empty()
                    #targetSprite.empty()
                    playerSprite.empty()
                    playerStepCapturerSprite.empty()
                    playerStepModulatorSprite.empty()

                    player = Player(currentTrial)
                    playerSprite.add(player)

                    if game_mode == "CP":
                        if currentTrial > 0:
                            block = Targets.ContinuousTarget(targetPath, trialLength, max_target_velocity, target_physics_mode)
                            targetSprite.add(block)
                            chosenTarget.add(random.choice(targetSprite.sprites()))

                        elif currentTrial == 0:
                            block = Targets.CircularReachTarget()
                            targetSprite.add(block)
                            chosenTarget.add(random.choice(targetSprite.sprites()))

                    elif game_mode == "DT_rect":

                        chosen = random.choice(targetSprite.sprites())
                        chosenTarget.add(chosen)  # add to target sprite list

                        targetSprite.remove(chosen)


                    stepcapturer = PlayerStepCapturer()
                    playerStepCapturerSprite.add(stepcapturer)
                    stepmodulator = PlayerStepModulator()
                    playerStepModulatorSprite.add(stepmodulator)

                    chosenTarget.draw(screen)

                    if experiment_mode:
                        random.shuffle(velocity_grid)
                        try:
                            if currentTrial > 0:
                                current_tvcv = velocity_grid.pop()
                                tvcv_order.append(current_tvcv)
                            elif currentTrial == 0:
                                current_tvcv = [0, 25]
                        except IndexError:
                            print("IndexError")
                            pass
                        for s in targetSprite.sprites():
                            s.max_target_velocity = current_tvcv[0]
                        for s in playerSprite.sprites():
                            s.max_cursor_velocity = current_tvcv[1]

                    frames = 0

                    start = True

            # if event.type == seconds_out:

        if start:
            frames += 1
            if currentTrial > 0:
                targetPath, targetVel = chosenTarget.update(frames, trial_no, target_no, targetPath, targetVel)
                v_step, trueVel = playerStepCapturerSprite.update(frames, trial_no, target_no, trueVel)
                v_modulated, modulation_gain = playerStepModulatorSprite.update(v_step)
                playerPath, playerVel = playerSprite.update(v_modulated, modulation_gain, frames, trial_no, target_no, playerPath, playerVel)
                targetNumber[trial_no][target_no].append(target_no)

                chosenTarget.clear(screen, background)
                playerSprite.clear(screen, background)  # order is important for transparency
                playerStepCapturerSprite.clear(screen, background)
                playerStepModulatorSprite.clear(screen, background)

                chosenTarget.draw(screen)
                playerSprite.draw(screen)
                playerStepCapturerSprite.draw(screen)
                playerStepModulatorSprite.draw(screen)

                if game_mode == "DT_rect":
                    if pygame.sprite.groupcollide(playerSprite, chosenTarget, False, False):
                        try:
                            for s in chosenTarget.sprites():
                                chosenTarget.remove(s)
                            # for s in playerSprite.sprites():
                            #     playerPath.append([s.report_trajectory()[0], s.report_trajectory()[1]])
                            # print(len(playerPath))

                            target_no += 1

                            chosen = random.choice(targetSprite.sprites())

                            current_chosen = str(chosen).split()[0].split("<")[1]
                            print(current_chosen)
                            print(current_chosen in ["BarTargetLeft", "BarTargetRight"])
                            if current_chosen in ["BarTargetLeft", "BarTargetRight"]:
                                while current_chosen == last_chosen_LR:
                                    print("A", current_chosen, last_chosen_LR)
                                    chosen = random.choice(targetSprite.sprites())
                                    current_chosen = str(chosen).split()[0].split("<")[1]
                                    if len(targetSprite.sprites()) <= 2:
                                        print('hi')
                                        break
                            elif current_chosen in ["BarTargetDown", "BarTargetUp"]:
                                while current_chosen == last_chosen_DU:
                                    print("B", current_chosen, last_chosen_DU)
                                    chosen = random.choice(targetSprite.sprites())
                                    current_chosen = str(chosen).split()[0].split("<")[1]
                                    if len(targetSprite.sprites()) <= 2:
                                        print('hi')
                                        break

                            chosenTarget.add(chosen)  # add to target sprite list

                            last_chosen = str(chosen).split()[0].split("<")[1]
                            if last_chosen in ["BarTargetLeft", "BarTargetRight"]:
                                last_chosen_LR = last_chosen
                            elif last_chosen in ["BarTargetDown", "BarTargetUp"]:
                                last_chosen_DU = last_chosen

                            targetSprite.remove(chosen)

                            #chosenTarget.update(frames, trial_no, target_no, targetPath, targetVel)  # unlike CP, we update bar targets even when currentTrial == 0  <- BAD APPROACH! game mode = "DT_rect" should just start out with +1 currentTrial to skip the calibration phase.
                            score += 1
                            # print(score)
                            screen.fill(pygame.Color("white"))

                            # Reset player
                            pygame.mouse.set_pos(375, 375)
                            playerSprite.empty()
                            playerStepCapturerSprite.empty()
                            playerStepModulatorSprite.empty()
                            pygame.time.wait(1000)
                            player = Player(currentTrial)
                            playerSprite.add(player)
                            stepcapturer = PlayerStepCapturer()
                            playerStepCapturerSprite.add(stepcapturer)
                            stepmodulator = PlayerStepModulator()
                            playerStepModulatorSprite.add(stepmodulator)
                        except IndexError:
                            pass

            elif currentTrial == 0:  # CP calibration trial
                if pygame.sprite.groupcollide(playerSprite, chosenTarget, False, True):
                    print("collision detected")
                    # explosionSprites.add(EnemyExplosion(self.rect.center))
                    if game_mode == "CP":
                        block = Targets.CircularReachTarget()
                        targetSprite.add(block)
                        chosenTarget.add(
                            random.choice(targetSprite.sprites()))  # replenish the chosenTarget single sprite group.

                    score += 1
                    # print(score)
                    screen.fill(pygame.Color("white"))

                    # Reset player
                    pygame.mouse.set_pos(375, 375)
                    playerSprite.empty()
                    playerStepCapturerSprite.empty()
                    playerStepModulatorSprite.empty()
                    player = Player(currentTrial)
                    playerSprite.add(player)
                    stepcapturer = PlayerStepCapturer()
                    playerStepCapturerSprite.add(stepcapturer)
                    stepmodulator = PlayerStepModulator()
                    playerStepModulatorSprite.add(stepmodulator)

                    # pygame.time.wait(3000) # delay between reaches?


            # print("input: ")
            # print(v_step)

            # if currentTrial > 0:
            #     vis_true_x_over_time.append(v_step[0])
            #     vis_true_y_over_time.append(v_step[1])

            # print(frames)
            if currentTrial > 0:  # if not a calibration trial
                if game_mode == "CP":
                    if frames == trialLength:  # after a trial

                        print("Just finished trial number " + str(currentTrial))
                        end = time.time()
                        print(str(end - starto) + ' seconds taken.')

                        if summary_visualization_mode:
                            playerSprite.sprites()[0].plot_error_vs_angle(bins=12)

                        currentTrial += 1
                        trial_no += 1
                        wait_frames = 0
                        frames = 0
                        start = False

                elif game_mode == "DT_rect":  # For DT_rect, unlike CP where trial length is defined by time,
                    if not chosenTarget.sprites():  # if rect list is empty
                        print("All sprites killed")
                        print("Just finished trial number " + str(currentTrial))
                        end = time.time()
                        print(str(end - starto) + ' seconds taken.')

                        currentTrial += 1
                        wait_frames = 0
                        frames = 0
                        start = False

            elif currentTrial == 0:
                if frames >= trialLength:
                    frames = trialLength - 1   # force frames to stay below triallength to circumvent data saving issue; since we don't save data for calibration trial anyway.
                if score == 5:   # Number of calibration targets needed to hit

                    currentTrial += 1
                    wait_frames = 0
                    frames = 0
                    start = False

            if currentTrial > trialCount:
                print("Experiment finished.")

                export_file_path = filedialog.askdirectory(title='Please choose a directory that you can access...')

                for trial_no in range(trialCount):
                    print("Now processing: " + str(trial_no))

                    print([j[0] for j in sum(playerPath[trial_no], [])])
                    print(len([j[0] for j in sum(playerPath[trial_no], [])]))

                    print([j for j in sum(targetNumber[trial_no], [])])
                    print(len([j for j in sum(targetNumber[trial_no], [])]))
                    trajectories = {
                        'playerPathTrialX': [j[0] for j in sum(playerPath[trial_no], [])],
                        'playerPathTrialY': [j[1] for j in sum(playerPath[trial_no], [])],
                        'targetPathTrialX': [j[0] for j in sum(targetPath[trial_no], [])],
                        'targetPathTrialY': [j[1] for j in sum(targetPath[trial_no], [])],
                        'playerVelTrialX': [j[0] for j in sum(playerVel[trial_no], [])],
                        'playerVelTrialY': [j[1] for j in sum(playerVel[trial_no], [])],
                        'targetVelTrialX': [j[0] for j in sum(targetVel[trial_no], [])],
                        'targetVelTrialY': [j[1] for j in sum(targetVel[trial_no], [])],
                        'trueVelTrialX': [j[0] for j in sum(trueVel[trial_no], [])],
                        'trueVelTrialY': [j[1] for j in sum(trueVel[trial_no], [])],
                        'targetNumber': [j for j in sum(targetNumber[trial_no], [])]
                    }
                    saved_trajectory_df = pd.DataFrame(trajectories, columns=list(trajectories.keys()))
                    if experiment_mode:
                        filename = export_file_path+'/'+'trial'+str(trial_no)+'target'+str(tvcv_order[trial_no][0])+'cursor'+str(tvcv_order[trial_no][1])+'-'+datetime_of_trials[trial_no+1]+'.csv'
                        saved_trajectory_df.to_csv(filename, index=False, header=True)
                        encrypt_data(filename=filename, key="J64ZHFpCWFlS9zT7y5zxuQN1Gb09y7cucne_EhuWyDM=")
                    if not experiment_mode:
                        saved_trajectory_df.to_csv(export_file_path+'/'+str(trial_no)+'.csv', index=False, header=True)

                    # From saved trajectory information, calculate tracking correlation coefficient
                    # x_corr = np.corrcoef(trajectories['PlayerPathX'], trajectories['TargetPathX'])[1, 0]
                    # x_corr_trials.append(x_corr)
                    # y_corr = np.corrcoef(trajectories['PlayerPathY'], trajectories['TargetPathY'])[1, 0]
                    # y_corr_trials.append(y_corr)

                if summary_visualization_mode:
                    plt.figure()
                    plt.hist(vis_xcontrol_b4_dwell, bins=50)
                    plt.title('Distribution of normalized x-dir control signals')
                    plt.xlabel('x-dir control signal')
                    plt.ylabel('Frequency')
                    plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
                    plt.axvline(x=d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.axvline(x=-d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.show()

                    plt.figure()
                    plt.hist(vis_ycontrol_b4_dwell, bins=50)
                    plt.title('Distribution of normalized y-dir control signals')
                    plt.xlabel('y-dir control signal')
                    plt.ylabel('Frequency')
                    plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
                    plt.axvline(x=d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.axvline(x=-d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.show()

                    plt.figure()
                    plt.hist(vis_xcontrol_dwell, bins=50)
                    plt.title('Distribution of normalized x-dir control signals after dwell check')
                    plt.xlabel('x-dir control signal')
                    plt.ylabel('Frequency')
                    plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
                    plt.axvline(x=d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.axvline(x=-d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.show()

                    plt.figure()
                    plt.hist(vis_ycontrol_dwell, bins=50)
                    plt.title('Distribution of normalized y-dir control signals after dwell check')
                    plt.xlabel('y-dir control signal')
                    plt.ylabel('Frequency')
                    plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
                    plt.axvline(x=d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.axvline(x=-d, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.show()

                    plt.figure()
                    plt.hist(vis_vnorms_b4_cap, bins=50)
                    plt.title('Distribution of velocities before max velocity enforcement, with dwell states')
                    plt.xlabel('velocity (pixels/frame)')
                    plt.ylabel('Frequency')
                    plt.axvline(x=max_cursor_velocity, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.show()

                    plt.figure()
                    plt.hist(vis_vnorms_capped, bins=50)
                    plt.title('Distribution of velocities after max velocity enforcement, with dwell states')
                    plt.xlabel('velocity (pixels/frame)')
                    plt.ylabel('Frequency')
                    plt.axvline(x=max_cursor_velocity, ymin=0, ymax=1000, color='red', linestyle='dashed')
                    plt.show()

                # np.set_printoptions(threshold=10)
                keepGoing = False

        # # See if the player block has collided with anything.
        #
        # targets_hit_list = pygame.sprite.spritecollide(player, chosenTarget, True)
        #
        # for target in targets_hit_list:
        #     score += 1
        #     print(score)

        pygame.display.flip()

    pygame.mouse.set_visible(True)  # return mouse
    print(tvcv_order)

if __name__ == "__main__":
    main()



