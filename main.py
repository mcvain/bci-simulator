# Closed-loop brain-computer interface simulation
# Feb 2022

from ui.get_params import *
from eeg.signal_source import *
from eeg.velocity_mapping import *
from decode.spatial_filter import *
from decode.decoder_arpsd import *
import pygame, pygame.gfxdraw, pygame.font, random, os, ctypes, tkinter, sys
from pygame.locals import *
from tkinter import messagebox
from datetime import datetime
from scipy import io
import numpy as np
import pandas as pd

score = 0
eeg_output_for_vis = np.empty([33, 1])  # 32 channels being generated. + 1 channel containing targetcode
params = launcher("subj")

max_target_velocity = params['max_target_velocity']
max_cursor_velocity = params['max_cursor_velocity']
trial_length = params['trial_length']
target_count = params['trial_count']
normalize_mode = params['normalize_mode']
game_mode = params['game_mode']
leadfield_mode = params['leadfield_mode']
target_physics_mode = params['target_physics_mode']
save_file_directory = params['save_file_directory']
experimental_param_array = params['experimental_param_array']
run_count = params['run_count']

if save_file_directory.get() == "":
    # hide the main tkinter window
    root = tkinter.Tk()
    root.attributes('-topmost', 1)
    root.withdraw()

    # invalid directory
    tkinter.messagebox.showinfo("Warning",
                                "You did not specify a save file directory, or the directory is inaccessible. Please restart the program!")

    root.deiconify()
    root.attributes("-topmost", True)
    sys.exit()

experimental_param_array = experimental_param_array.split(" ")

full_screen_mode = True
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (515, 100)  # location of window
pygame.init()
myfont = pygame.font.SysFont('Calibri', 20)
textsurf0 = myfont.render('Waiting for next trial...', False, (0, 0, 0))
textsurf1 = myfont.render('Next trial ready.', False, (0, 0, 0))
textsurf2 = myfont.render('Please return mouse to center of desk and press the spacebar when ready to start.',
                          False, (0, 0, 0))

# screen setup
screen_width = 800
screen_height = 800

if full_screen_mode:
    flags = FULLSCREEN | HWSURFACE | HWACCEL | DOUBLEBUF
else:
    flags = HWSURFACE | HWACCEL | DOUBLEBUF

ctypes.windll.user32.SetProcessDPIAware()
# sysres = (ctypes.windll.user32.GetSystemMetrics(0),ctypes.windll.user32.GetSystemMetrics(1))
screen = pygame.display.set_mode((screen_width, screen_height), flags)
screen.set_alpha(None)

# sprite group initialization
playerSprite = pygame.sprite.Group()
chosenTarget = pygame.sprite.GroupSingle()
wrongTarget = pygame.sprite.GroupSingle()
playerStepCapturerSprite = pygame.sprite.GroupSingle()
playerStepModulatorSprite = pygame.sprite.GroupSingle()

# initialize data to be saved
player_path_x = [[[] for tr in range(target_count)] for run in range(run_count)]
player_path_y = [[[] for tr in range(target_count)] for run in range(run_count)]
true_vel_x = [[[] for tr in range(target_count)] for run in range(run_count)]
true_vel_y = [[[] for tr in range(target_count)] for run in range(run_count)]
correct_targets = [[[] for tr in range(target_count)] for run in range(run_count)]
input_targets = [[[] for tr in range(target_count)] for run in range(run_count)]
datetimes = [[[] for tr in range(target_count)] for run in range(run_count)]

# atlas and leadfield configuration
import_lf_within_source_signals(leadfield_mode)

# import atlas
if leadfield_mode == 'sereega':
    M1_left, M1_right = import_atlas('data/atlas/brodmann_area_def.mat', 'SEREEGA')
elif leadfield_mode == 'brainstorm':
    M1_left, M1_right = import_atlas('data/atlas/brainnetome_area_def.mat', 'brainstorm')

# epoch configuration
srate, epochLength, samples = epoch_config(srate=512, epochLength=33)

# import leadfield
if leadfield_mode == 'sereega':
    leadfield, lf, orientation, pos, chanlocs = import_leadfield('data/leadfield/32_channel_nyhead_lf.mat', 'sereega')
elif leadfield_mode == 'brainstorm':
    leadfield, lf, orientation, pos, chanlocs = import_leadfield('data/leadfield/openmeeg_fsaverage_lf.mat',
                                                                 'brainstorm')

# import channel labels and choose channels for spatial Laplacian filter
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

# prepare background noise signal
noise = generate_noise_signal('brown-unif', 5.0)

# prepare background/resting components
if leadfield_mode == 'sereega':
    component_list_background = create_component('random', 500, noise, component_list=[], absolute_mode=False,
                                                 mode='sereega')
elif leadfield_mode == 'brainstorm':
    component_list_background = create_component('random', 500, noise, component_list=[], absolute_mode=False,
                                                 mode='brainstorm')

pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])

run_no = 0
target_no = 0

max_feedback_period = int(6 * 30)  # 6*30 frames


class ActiveTarget(pygame.sprite.Sprite):
    def __init__(self, dir):
        pygame.sprite.Sprite.__init__(self)
        self.dir = dir

        if self.dir == 'up' or self.dir == 'down':
            self.target_width = 700
            self.target_height = 25
        elif self.dir == 'left' or self.dir == 'right':
            self.target_width = 25
            self.target_height = 700

        self.image = pygame.Surface((self.target_width, self.target_height))
        self.image.fill((255, 255, 255))

        self.rect = self.image.get_rect()

    def update(self, frames, run_no, target_no):
        global screen_width, screen_height

        if self.dir == 'up':
            self.target_width = 700
            self.target_height = 25
            self.rect.x = (screen_width - self.target_width) // 2
            self.rect.y = 25
        elif self.dir == 'down':
            self.target_width = 700
            self.target_height = 25
            self.rect.x = (screen_width - self.target_width) // 2
            self.rect.y = screen_height - 50
        elif self.dir == 'left':
            self.target_width = 25
            self.target_height = 700
            self.rect.x = 25
            self.rect.y = (screen_height - self.target_height) // 2
        elif self.dir == 'right':
            self.target_width = 25
            self.target_height = 700
            self.rect.x = screen_width - 50
            self.rect.y = (screen_height - self.target_height) // 2

        pygame.draw.rect(self.image, (255, 0, 0), (0, 0, 700, 700), 0)


class InactiveTarget(pygame.sprite.Sprite):
    def __init__(self, dir):
        pygame.sprite.Sprite.__init__(self)
        self.dir = dir

        if self.dir == 'up' or self.dir == 'down':
            self.target_width = 700
            self.target_height = 25
        elif self.dir == 'left' or self.dir == 'right':
            self.target_width = 25
            self.target_height = 700

        self.image = pygame.Surface((self.target_width, self.target_height))
        self.image.fill((255, 255, 255))

        self.rect = self.image.get_rect()

    def update(self, frames, run_no, target_no):
        global score, screen_width, screen_height
        if self.dir == 'up':
            self.target_width = 700
            self.target_height = 25
            self.rect.x = (screen_width - self.target_width) // 2
            self.rect.y = 25
        elif self.dir == 'down':
            self.target_width = 700
            self.target_height = 25
            self.rect.x = (screen_width - self.target_width) // 2
            self.rect.y = screen_height - 50
        elif self.dir == 'left':
            self.target_width = 25
            self.target_height = 700
            self.rect.x = 25
            self.rect.y = (screen_height - self.target_height) // 2
        elif self.dir == 'right':
            self.target_width = 25
            self.target_height = 700
            self.rect.x = screen_width - 50
            self.rect.y = (screen_height - self.target_height) // 2


class PlayerStepCapturer(pygame.sprite.Sprite):
    def __init__(self):
        global screen_width, screen_height
        pygame.sprite.Sprite.__init__(self)
        self.radius = int(screen_width * 0.1 * 0.5)  # 10% of screen width, halved because radius.

        self.image = pygame.Surface((self.radius * 2, self.radius * 2))
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))

        self.rect = self.image.get_rect()

        self.rect.x = screen_width // 2 - self.radius
        self.rect.y = screen_height // 2 - self.radius

        self.pos_t1 = (screen_width // 2 - self.radius, screen_height // 2 - self.radius)
        self.pos_t0 = (screen_width // 2 - self.radius, screen_height // 2 - self.radius)

    def update(self, frames, run_no, target_no, true_vel_x, true_vel_y):
        self.pos_t1 = pygame.mouse.get_pos()

        step_velocity_vector = (self.pos_t1[0] - self.pos_t0[0], self.pos_t1[1] - self.pos_t0[1])

        self.rect.x = self.pos_t1[0]
        self.rect.y = self.pos_t1[1]

        self.pos_t1 = (screen_width // 2 - self.radius, screen_height // 2 - self.radius)  # reset back to midpoint
        pygame.mouse.set_pos(self.pos_t1)

        true_vel_x[run_no][target_no].append(step_velocity_vector[0])
        true_vel_y[run_no][target_no].append(step_velocity_vector[1])

        return step_velocity_vector, true_vel_x, true_vel_y


class PlayerCursor(pygame.sprite.Sprite):
    def __init__(self):
        global run_no, experimental_param_array
        pygame.sprite.Sprite.__init__(self)

        self.radius = int(screen_width * 0.1 * 0.5)
        self.edge_threshold = 0.01
        self.image = pygame.Surface((self.radius * 2, self.radius * 2))
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))

        pygame.draw.circle(self.image, (0, 0, 255), [self.radius, self.radius], self.radius)
        self.rect = self.image.get_rect()
        self.rect.x = screen_width // 2 - self.radius
        self.rect.y = screen_height // 2 - self.radius

        # build position tracker since rect does not support floats
        self.current_pos = [float(self.rect.x), float(self.rect.y)]

        self.v = pygame.Vector2()
        self.v.xy = 0.0001, 0.0001  # avoid division by zero
        self.gain = 5  # gain of the sigmoid function

        if experimental_param_array[run_no].startswith("CS"):
            max_cursor_velocity = int(experimental_param_array[run_no].split("CS")[1])  # in pixels/s
            max_cursor_velocity = max_cursor_velocity / 30  # convert units
        else:  # default
            max_cursor_velocity = 5.5
        self.max_cursor_velocity = max_cursor_velocity

    def update(self, step_velocity_vector, x_control_buffer, y_control_buffer, targetcode):
        global game_mode, eeg_output_for_vis
        v = step_velocity_vector
        amp_C3, amp_C4, scaling_factor = velocity_to_mod_amplitude(v[0], v[1], 'perturbed', self.gain)

        freqRange = [3, 5, 12, 14]
        target_C3_x = modulate_signal(freqRange, 5, 'invburst',
                                      5, 20, 0.5, amp_C3)
        target_C3_y = modulate_signal(freqRange, 5, 'invburst',
                                      5, 20, 0.5, amp_C3)
        target_C4_x = modulate_signal(freqRange, 5, 'invburst',
                                      5, 20, 0.5, amp_C4)
        target_C4_y = modulate_signal(freqRange, 5, 'invburst',
                                      5, 20, 0.5, amp_C4)

        # Create components
        active_component_list = component_list_background[:]  # copy over background components
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

        print(M1_left)
        print(M1_right)

        # Generate scalp data
        eeg = generate_scalp_data(active_component_list)
        if targetcode == 'left':
            targetcode = 2
        elif targetcode == 'right':
            targetcode = 1
        targetcodes = np.full((1, eeg.shape[1]), targetcode)
        eeg_with_targetcode = np.vstack((eeg, targetcodes))
        eeg_output_for_vis = np.hstack((eeg_output_for_vis, eeg_with_targetcode))
        # Spatial filtering and decoding
        eeg_spatial_filtered = spatial_filter(eeg, channelSelection)
        if experimental_param_array[run_no].startswith("BW"):
            buffer_length = int(experimental_param_array[run_no].split("BW")[1])  # in seconds
            buffer_length = buffer_length * 30  # in frames
        else:  # default
            buffer_length = 900  # 900 frames = 30 seconds.

        x_control, y_control, x_control_buffer, y_control_buffer = decoder_arpsd(eeg_spatial_filtered,
                                                                                 buffer_length,
                                                                                 x_control_buffer,
                                                                                 y_control_buffer,
                                                                                 normalize_mode=normalize_mode)
        # restore scaling to the control vectors
        x_control = x_control * scaling_factor
        y_control = y_control * scaling_factor

        # velocity-based approach
        self.v.xy = x_control, y_control

        # screen edge collision check
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

        # threshold velocity
        v = np.array([[self.v.x], [self.v.y]], dtype=np.float32)
        if np.linalg.norm(v) > self.max_cursor_velocity:
            control_max = np.array([[self.max_cursor_velocity], [self.max_cursor_velocity]], dtype=np.float32)
            new_v = control_max * v.transpose() / np.linalg.norm(v)
            self.v.x = new_v[0][0]
            self.v.y = new_v[0][1]

        # update position
        self.current_pos[0] += self.v.x
        self.current_pos[1] += self.v.y

        self.rect.x = int(self.current_pos[0])
        if game_mode != 'DT_1D':  # disable horizontal movement for discrete LR
            self.rect.y = int(self.current_pos[1])

        player_path_x[run_no][target_no].append(self.rect.x)
        player_path_y[run_no][target_no].append(self.rect.y)

        return player_path_x, player_path_y, x_control_buffer, y_control_buffer


def main():
    global score, run_no, target_no, player_path_x, player_path_y, true_vel_x, true_vel_y, correct_targets, input_targets, datetimes, eeg_output_for_vis
    frames = 0
    wait_frames = 0
    x_control_buffer = []
    y_control_buffer = []
    pygame.display.set_caption("BCI Simulator")

    # configure the background
    background = pygame.Surface(screen.get_size())
    background.fill((255, 255, 255))
    screen.blit(background, (0, 0))
    pygame.mouse.set_visible(False)  # hide mouse
    clock = pygame.time.Clock()

    # status variables
    keepGoing = True  # controls whether the program continues to run
    start = False  # controls the feedback control period
    seconds_out_tracker = False  # controls the break period

    # initialize trial number tracker
    if game_mode == "CP":
        run_no = -1
    else:
        run_no = 0

    seconds_out = pygame.USEREVENT + 1  # creates "break time over" event 

    # Main loop
    while keepGoing:
        clock.tick(30)

        if not start:
            chosenTarget.clear(screen, background)
            playerSprite.clear(screen, background)
            seconds_out_tracker = False
            if run_no > 0:
                screen.blit(textsurf0, (200, 400))

            wait_frames += 1  # countdown to next trial

            if wait_frames > 100:
                screen.fill(pygame.Color("white"))
                screen.blit(textsurf1, (282, 400))
                screen.blit(textsurf2, (20, 425))

                my_event = pygame.event.Event(seconds_out, {"message": "Break time over"})
                pygame.event.post(my_event)
                seconds_out_tracker = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    # Force quit by pressing 'P'
                    keepGoing = False

                if event.key == pygame.K_SPACE and not start and seconds_out_tracker:
                    targetSprite = pygame.sprite.LayeredUpdates()  # list of target sprites
                    inactivetargetSprite = pygame.sprite.LayeredUpdates()
                    if game_mode == "DT_2D":
                        for n in range(target_count / 4):
                            block = ActiveTarget('left')
                            targetSprite.add(block)
                            block = ActiveTarget('right')
                            targetSprite.add(block)
                            block = ActiveTarget('up')
                            targetSprite.add(block)
                            block = ActiveTarget('down')
                            targetSprite.add(block)
                    elif game_mode == "DT_1D":
                        correct_order = []
                        incorrect_order = []
                        for n in range(int(target_count / 2)):  # Blockwise randomize
                            print(n)
                            eitheror = ['left', 'right']
                            first_in_block = random.choice(eitheror)

                            correct_targets[run_no][n * 2] = first_in_block
                            eitheror.remove(first_in_block)
                            second_in_block = eitheror[0]

                            correct_targets[run_no][n * 2 + 1] = second_in_block

                            correct_order.append(first_in_block)
                            correct_order.append(second_in_block)
                            incorrect_order.append(second_in_block)
                            incorrect_order.append(first_in_block)

                            block = ActiveTarget(first_in_block)
                            targetSprite.add(block)
                            block = ActiveTarget(second_in_block)
                            targetSprite.add(block)

                            inactiveblock = InactiveTarget(second_in_block)
                            inactivetargetSprite.add(inactiveblock)
                            inactiveblock = InactiveTarget(first_in_block)
                            inactivetargetSprite.add(inactiveblock)
                        print(correct_order)
                        print(incorrect_order)

                    screen.fill(pygame.Color("white"))
                    playerSprite.clear(screen, background)
                    chosenTarget.clear(screen, background)
                    playerStepCapturerSprite.clear(screen, background)

                    # Reset after an iteration
                    chosenTarget.empty()
                    playerSprite.empty()
                    playerStepCapturerSprite.empty()

                    player = PlayerCursor()
                    playerSprite.add(player)

                    if game_mode == "DT_1D" or game_mode == "DT_2D":
                        chosen = targetSprite.get_sprite(target_no)
                        wrong = inactivetargetSprite.get_sprite(target_no)
                        chosenTarget.empty()
                        chosenTarget.add(chosen)  # add to current activated target sprite
                        wrongTarget.empty()
                        wrongTarget.add(wrong)

                    elif game_mode == "CP":  # NOT SUPPORTED IN THIS BUILD
                        if run_no >= 0:
                            block = Targets.ContinuousTarget(targetPath, trialLength, max_target_velocity,
                                                             target_physics_mode)
                            targetSprite.add(block)
                            chosenTarget.add(random.choice(targetSprite.sprites()))

                        elif run_no == -1:
                            block = Targets.CircularReachTarget()
                            targetSprite.add(block)
                            chosenTarget.add(random.choice(targetSprite.sprites()))

                    stepcapturer = PlayerStepCapturer()
                    playerStepCapturerSprite.add(stepcapturer)
                    screen.fill(pygame.Color("white"))
                    chosenTarget.update(frames, run_no, target_no)
                    chosenTarget.draw(screen)
                    pygame.display.update()
                    pygame.time.wait(2000)

                    frames = 0
                    start = True

        if start:
            frames += 1
            if run_no >= 0:
                chosenTarget.update(frames, run_no, target_no)
                wrongTarget.update(frames, run_no, target_no)
                step_velocity_vector, true_vel_x, true_vel_y = playerStepCapturerSprite.update(frames, run_no,
                                                                                               target_no, true_vel_x,
                                                                                               true_vel_y)
                player_path_x, player_path_y, x_control_buffer, y_control_buffer = playerSprite.update(
                    step_velocity_vector, x_control_buffer, y_control_buffer, correct_order[target_no])

                chosenTarget.clear(screen, background)
                playerSprite.clear(screen, background)
                playerStepCapturerSprite.clear(screen, background)

                chosenTarget.draw(screen)
                if frames == 1:  # shows the target to aim for before allowing control of cursor.
                    pygame.display.update()
                    pygame.time.wait(2000)
                playerSprite.draw(screen)
                playerStepCapturerSprite.draw(screen)

                if game_mode == "DT_1D" or game_mode == "DT_2D":
                    if pygame.sprite.groupcollide(playerSprite, chosenTarget, False, False):
                        io.savemat('eeg_output_for_vis.mat', {'mydata': eeg_output_for_vis})
                        input_targets[run_no][target_no] = correct_order[target_no]
                        now = datetime.now()
                        now = now.strftime("%Y-%m-%d-%H-%M-%S")
                        datetimes[run_no][target_no] = now

                        try:  # if this is not the last target in this run
                            target_no += 1
                            chosen = targetSprite.get_sprite(target_no)
                            wrong = inactivetargetSprite.get_sprite(target_no)
                            chosenTarget.empty()
                            chosenTarget.add(chosen)  # add to current activated target sprite
                            wrongTarget.empty()
                            wrongTarget.add(wrong)
                        except:
                            pass

                        screen.fill(pygame.Color("white"))

                        # reset player
                        pygame.mouse.set_pos(400, 400)
                        playerSprite.empty()
                        playerStepCapturerSprite.empty()
                        pygame.time.wait(1000)
                        player = PlayerCursor()
                        playerSprite.add(player)
                        stepcapturer = PlayerStepCapturer()
                        playerStepCapturerSprite.add(stepcapturer)
                        frames = 0

                    if pygame.sprite.groupcollide(playerSprite, wrongTarget, False, False):
                        io.savemat('eeg_output_for_vis.mat', {'mydata': eeg_output_for_vis})
                        input_targets[run_no][target_no] = incorrect_order[target_no]
                        now = datetime.now()
                        now = now.strftime("%Y-%m-%d-%H-%M-%S")
                        datetimes[run_no][target_no] = now

                        try:  # if this is not the last target in this run
                            target_no += 1
                            chosen = targetSprite.get_sprite(target_no)
                            wrong = inactivetargetSprite.get_sprite(target_no)
                            chosenTarget.empty()
                            chosenTarget.add(chosen)  # add to current activated target sprite
                            wrongTarget.empty()
                            wrongTarget.add(wrong)
                        except:
                            pass

                        screen.fill(pygame.Color("white"))

                        # reset player
                        pygame.mouse.set_pos(400, 400)
                        playerSprite.empty()
                        playerStepCapturerSprite.empty()
                        pygame.time.wait(1000)
                        player = PlayerCursor()
                        playerSprite.add(player)
                        stepcapturer = PlayerStepCapturer()
                        playerStepCapturerSprite.add(stepcapturer)
                        frames = 0

                    if frames == max_feedback_period:
                        io.savemat('eeg_output_for_vis.mat', {'mydata': eeg_output_for_vis})
                        input_targets[run_no][target_no] = "timeout"
                        now = datetime.now()
                        now = now.strftime("%Y-%m-%d-%H-%M-%S")
                        datetimes[run_no][target_no] = now

                        try:  # if this is not the last target in this run
                            target_no += 1
                            chosen = targetSprite.get_sprite(target_no)
                            wrong = inactivetargetSprite.get_sprite(target_no)
                            chosenTarget.empty()
                            chosenTarget.add(chosen)  # add to current activated target sprite
                            wrongTarget.empty()
                            wrongTarget.add(wrong)
                        except:
                            pass

                        screen.fill(pygame.Color("white"))

                        # Reset player
                        pygame.mouse.set_pos(400, 400)
                        playerSprite.empty()
                        playerStepCapturerSprite.empty()
                        pygame.time.wait(1000)
                        player = PlayerCursor()
                        playerSprite.add(player)
                        stepcapturer = PlayerStepCapturer()
                        playerStepCapturerSprite.add(stepcapturer)
                        frames = 0

            elif run_no == -1:
                # CP calibration trial. not included in this build.
                pass

            if game_mode == "DT_1D" or game_mode == "DT_2D":
                if target_no == target_count:
                    print("All sprites killed")
                    print("Just finished run number " + str(run_no))

                    if experimental_param_array[run_no].startswith("CO"):
                        carryon = int(experimental_param_array[run_no].split("CO")[1])
                        if carryon == 1 or carryon == 2:
                            pass
                        elif carryon == 0:
                            x_control_buffer = []
                            y_control_buffer = []
                        else:  # default
                            x_control_buffer = []
                            y_control_buffer = []

                    run_no += 1
                    target_no = 0
                    wait_frames = 0
                    frames = 0
                    start = False

            elif game_mode == "CP" and run_no == -1:
                if frames >= trial_length:
                    frames = trial_length - 1  # force frames to stay below trial_length
                if score == 5:  # number of calibration targets needed to hit
                    run_no += 1
                    wait_frames = 0
                    frames = 0
                    start = False
            if run_no == run_count:
                print("Experiment finished.")

                for r in range(run_count):
                    trial_data = {
                        'PlayerPathX': player_path_x[r],
                        'PlayerPathY': player_path_y[r],
                        'TrueVelX': true_vel_x[r],
                        'TrueVelY': true_vel_y[r],
                        'CorrectTarget': correct_targets[r],
                        'InputTarget': input_targets[r],
                        'DateTime': datetimes[r]
                    }

                    trial_data_df = pd.DataFrame(trial_data, columns=list(trial_data.keys()))
                    trial_data_df.to_csv(
                        save_file_directory.get() + '/' + str(r) + '_' + str(experimental_param_array[r]) + '.csv',
                        index=False, header=True)

                # hide the main tkinter window
                root = tkinter.Tk()
                root.attributes('-topmost', 1)
                root.withdraw()

                # results saved
                tkinter.messagebox.showinfo("Thank You",
                                            "The experiment is finished. Please find the data saved in " + str(
                                                save_file_directory.get()))
                root.deiconify()
                root.attributes("-topmost", True)
                keepGoing = False
        pygame.display.flip()
    pygame.mouse.set_visible(True)  # return mouse


if __name__ == "__main__":
    main()
