import pygame
import math
import numpy as np
import random

screen_width = 800
screen_height = 800

# Continuous pursuit
class ContinuousTarget(pygame.sprite.Sprite):
    def __init__(self, targetPath, trialLength, max_target_velocity, target_physics_mode):
        global screen_width, screen_height
        pygame.sprite.Sprite.__init__(self)
        self.radius = 25
        self.image = pygame.Surface((self.radius * 2, self.radius * 2))
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))

        pygame.draw.circle(self.image, (255, 0, 0), [self.radius, self.radius], self.radius)
        self.rect = self.image.get_rect()

        self.rect.x = screen_width // 2 - self.radius
        self.rect.y = screen_height // 2 - self.radius

        self.m = 280
        self.v = pygame.Vector2()
        self.v.xy = 0.0001, 0.0001  # Avoid division by zero

        self.f_external = pygame.Vector2()

        self.friction_coeff = 20
        self.f_friction = pygame.Vector2()
        self.drag_coeff = 0.3
        self.f_drag = pygame.Vector2()

        edge_threshold = 0.1
        self.edge_threshold = edge_threshold
        self.max_target_velocity = max_target_velocity

        self.physics_mode = target_physics_mode

        self.iteration = 0

    def update(self, frames, trial_no, target_no, targetPath, targetVel):
        # New approach (4/20/2021) - Transferred from bcisim_TargetKinemSim3.m
        speed_limit = self.max_target_velocity
        if self.physics_mode == "adjust_drag":
            self.drag_coeff = 43.675372 * math.exp(-0.2277747 * speed_limit)
        # print("max target velocity: " + str(speed_limit))
        # print("drag coefficient: " + str(self.drag_coeff))

        self.f_friction = -self.friction_coeff * (self.v / self.v.length())
        self.f_drag = -self.drag_coeff * self.v * self.v.length()
        current_term = self.v + self.f_drag/self.m + self.f_friction/self.m

        if self.physics_mode == "adjust_distribution":
            a = current_term.x; b = current_term.y
            mu = (-a, -b)
            sigma = [[(speed_limit/3)**2, 0], [0, (speed_limit/3)**2]]
            random_draw = np.random.multivariate_normal(mu, sigma, 1)
            self.f_external.xy = random_draw[0][0], random_draw[0][1]
            v_proposed = current_term + self.f_external
        elif self.physics_mode == "adjust_drag":  # Science Robotics paper
            mu = (0, 0)
            sigma = [[150000, 0], [0, 150000]]
            random_draw = np.random.multivariate_normal(mu, sigma, 1)
            self.f_external.xy = random_draw[0][0], random_draw[0][1]
            v_proposed = self.v + self.f_drag / self.m + self.f_friction / self.m + self.f_external / self.m  # this one includes /m

        # Pass the velocity into edge collision check:
        if self.rect.x + v_proposed.x <= (screen_width - self.radius*2) * self.edge_threshold:
            print("hit left wall")
            self.f_external.x = abs(self.f_external.x)
            self.f_drag.x = abs(self.f_drag.x)
            self.f_friction.x = abs(self.f_friction.x)
        elif self.rect.x + v_proposed.x >= (screen_width - self.radius*2) * (1 - self.edge_threshold):
            print("hit right wall")
            self.f_external.x = -abs(self.f_external.x)
            self.f_drag.x = -abs(self.f_drag.x)
            self.f_friction.x = -abs(self.f_friction.x)
        if self.rect.y + v_proposed.y <= (screen_height - self.radius*2) * self.edge_threshold:
            print("hit top wall")
            self.f_external.y = abs(self.f_external.y)
            self.f_drag.y = abs(self.f_drag.y)
            self.f_friction.y = abs(self.f_friction.y)
        elif self.rect.y + v_proposed.y >= (screen_height - self.radius*2) * (1 - self.edge_threshold):
            print("hit bottom wall")
            self.f_external.y = -abs(self.f_external.y)
            self.f_drag.y = -abs(self.f_drag.y)
            self.f_friction.y = -abs(self.f_friction.y)

        # Update the velocity according to reversed forces (if applicable).
        if self.physics_mode == "adjust_distribution":
            self.v = self.v + self.f_drag/self.m + self.f_friction/self.m + self.f_external
        elif self.physics_mode == "adjust_drag":
            self.v = self.v + self.f_drag/self.m + self.f_friction/self.m + self.f_external/self.m

        self.iteration += 1
        # Update cursor positions; despite the int(), the floating velocity values are preserved
        self.rect.x = self.rect.x + int(self.v.x)
        self.rect.y = self.rect.y + int(self.v.y)

        # Rig starting location
        if self.iteration == 1:
            start_offset = np.array([(-1 + 2*np.random.random_sample()), (-1 + 2*np.random.random_sample())], dtype=np.float32)
            start_offset = 125 * start_offset / np.linalg.norm(start_offset)  # start 125px away from center
            self.rect.x = screen_width // 2 - self.radius + start_offset[0]
            self.rect.y = screen_height // 2 - self.radius + start_offset[1]
            print("Starting position: " + str(self.rect.x) + ", " + str(self.rect.y))
            self.v.xy = 0.0001, 0.0001  # Same as init settings

        # Save position and velocity information
        targetPath[trial_no][target_no].append([self.rect.x, self.rect.y])
        targetVel[trial_no][target_no].append([self.v.x, self.v.y])

        return targetPath, targetVel


# Calibration target that goes in a circle (unused)
class CalibrationTarget(ContinuousTarget):
    def __init__(self):
        super().__init__()
        # The "center" the sprite will orbit
        self.center_x = screen_width // 2 - self.radius
        self.center_y = screen_height // 2 - self.radius

        # Current angle in radians
        self.angle = 0

        # How far away from the center to orbit, in pixels
        self.circle_radius = 50

        # How fast to orbit, in radians per frame
        self.speed = 0.2

    def update(self, frames):
        # Calculate a new x, y
        self.rect.x = self.circle_radius * math.sin(self.angle) + self.center_x
        self.rect.y = self.circle_radius * math.cos(self.angle) + self.center_y

        # Increase the angle in prep for the next round.
        self.angle += self.speed


score = 0


# Discrete circular center-out reach trials
class CircularReachTarget(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((50, 50))
        self.image.fill((255, 255, 255))
        self.image.set_colorkey((255, 255, 255))  # idk why this line doesnt work for transparency
        # pygame.draw.circle(self.image, (255, 0, 0), (25, 25), 25, 0)

        # The "center" the sprite will orbit
        self.center_x = 375
        self.center_y = 375

        # Current angle in radians
        self.angle = random.uniform(0, 2*math.pi)

        # How far away from the center to orbit, in pixels
        self.radius = 325

        # How fast to orbit, in radians per frame
        self.speed = 0

        pygame.draw.circle(self.image, (255, 0, 0), (25, 25), 25)
        self.rect = self.image.get_rect()
        self.rect.x = int(self.radius * math.sin(self.angle) + self.center_x)
        self.rect.y = int(self.radius * math.cos(self.angle) + self.center_y)


# Discrete 4-direction reach
class BarTarget(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)


class BarTargetUp(BarTarget):
    def __init__(self):
        BarTarget.__init__(self)
        self.target_loc = 3
        self.target_width = 700
        self.target_height = 25
        self.image = pygame.Surface((self.target_width, self.target_height))
        self.image.fill((255, 255, 255))
        # self.image.set_colorkey((255, 255, 255))  # idk why this line doesnt work for transparency
        # pygame.draw.circle(self.image, (255, 0, 0), (25, 25), 25, 0)

        self.rect = self.image.get_rect()

    def update(self, frames, trial_no, target_no, targetPath, targetVel):
        global score

        self.rect.x = (screen_width - self.target_width) // 2
        self.rect.y = 25  # screen_height - 100

        pygame.draw.rect(self.image, (255, 0, 0), (0, 0, 700, 700), 0)

        targetPath[trial_no][target_no].append([self.rect.x, self.rect.y])
        targetVel[trial_no][target_no].append([self.target_loc, self.target_loc])

        return targetPath, targetVel


class BarTargetDown(BarTarget):
    def __init__(self):
        BarTarget.__init__(self)
        self.target_loc = 4
        self.target_width = 700
        self.target_height = 25
        self.image = pygame.Surface((self.target_width, self.target_height))
        self.image.fill((255, 255, 255))
        # self.image.set_colorkey((255, 255, 255))  # idk why this line doesnt work for transparency
        # pygame.draw.circle(self.image, (255, 0, 0), (25, 25), 25, 0)

        self.rect = self.image.get_rect()

    def update(self, frames, trial_no, target_no, targetPath, targetVel):
        global score
        self.target_width = 700
        self.target_height = 25
        self.rect.x = (screen_width - self.target_width) // 2
        self.rect.y = screen_height - 50

        pygame.draw.rect(self.image, (255, 0, 0), (0, 0, 700, 700), 0)

        targetPath[trial_no][target_no].append([self.rect.x, self.rect.y])
        targetVel[trial_no][target_no].append([self.target_loc, self.target_loc])

        return targetPath, targetVel


class BarTargetLeft(BarTarget):
    def __init__(self):
        BarTarget.__init__(self)
        self.target_loc = 2
        self.target_width = 25
        self.target_height = 700
        self.image = pygame.Surface((self.target_width, self.target_height))
        self.image.fill((255, 255, 255))
        # self.image.set_colorkey((255, 255, 255))  # idk why this line doesnt work for transparency
        # pygame.draw.circle(self.image, (255, 0, 0), (25, 25), 25, 0)

        self.rect = self.image.get_rect()

    def update(self, frames, trial_no, target_no, targetPath, targetVel):
        global score
        self.target_width = 25
        self.target_height = 700
        self.rect.x = 25
        self.rect.y = (screen_height - self.target_height) // 2

        pygame.draw.rect(self.image, (255, 0, 0), (0, 0, 700, 700), 0)

        targetPath[trial_no][target_no].append([self.rect.x, self.rect.y])
        targetVel[trial_no][target_no].append([self.target_loc, self.target_loc])

        return targetPath, targetVel


class BarTargetRight(BarTarget):
    def __init__(self):
        BarTarget.__init__(self)
        self.target_loc = 1
        self.target_width = 25
        self.target_height = 700
        self.image = pygame.Surface((self.target_width, self.target_height))
        self.image.fill((255, 255, 255))
        # self.image.set_colorkey((255, 255, 255))  # idk why this line doesnt work for transparency
        # pygame.draw.circle(self.image, (255, 0, 0), (25, 25), 25, 0)

        self.rect = self.image.get_rect()

    def update(self, frames, trial_no, target_no, targetPath, targetVel):
        global score
        self.target_width = 25
        self.target_height = 700
        self.rect.x = screen_width - 50
        self.rect.y = (screen_height - self.target_height) // 2

        pygame.draw.rect(self.image, (255, 0, 0), (0, 0, 700, 700), 0)

        targetPath[trial_no][target_no].append([self.rect.x, self.rect.y])
        targetVel[trial_no][target_no].append([self.target_loc, self.target_loc])

        return targetPath, targetVel



