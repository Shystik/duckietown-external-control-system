#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import time

from PIL import Image
import argparse
import sys
import math
import random

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator as sim

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="Montreal_loop")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()


if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

points = [[0.99,0.00,0.45], #spawn
          [0.6,0.00,0.18], [0.33,0.00,0.23], [0.18,0.00,0.5], [0.18,0.00,0.6],
          [0.18,0.00,2.25], [0.23,0.00,2.6], [0.45,0.00,2.8],
          [0.5,0.00,2.8], [0.7, 0.0, 2.83],
          [0.7,0.00,2.9], [0.8,0.0,3.3], [1.1,0.00,3.4],
          [2.3,0.00,3.4],[2.7,0.0,3.3], [2.8,0.00,3.0],
          [2.8,0.00,0.7], [2.7,0.0,0.3], [2.5,0.00,0.18]]
global_i=1

def noise (cur_pos, cur_angel):
    for i in range(3):
        if i%2==0:
          cur_pos[i] = random.triangular(cur_pos[i]-0.02, cur_pos[i]+0.02)
    return (cur_pos, random.triangular(cur_angel-0.1, cur_angel+0.1))

class Delay:
    def __init__(self):
        self.step = 10  # количество шагов задержки
        self.queue_pos = []
        self.queue_ang = []
    def delay (self, cur_pos, cur_angle):
        self.queue_pos.append(cur_pos)
        self.queue_ang.append(cur_angle)
        if (len(self.queue_pos) == self.step):
            cur_pos = self.queue_pos.pop(0)
            cur_angle = self.queue_ang.pop(0)
            print(self.queue_pos[self.step-2], math.sqrt((self.queue_pos[self.step-2][0]-cur_pos[0])*(self.queue_pos[self.step-2][0]-cur_pos[0])
                                                +(self.queue_pos[self.step-2][2]-cur_pos[2])*(self.queue_pos[self.step-2][2]-cur_pos[2])))
        return (cur_pos, cur_angle)

delay = Delay()

def move_to(cur_pos, cur_angle):
    
    global global_i
    if (cur_angle < 0): #Перевод в окружность от 0 до 2 Пи
        cur_angle += 2*math.pi
    cur_pos, cur_angel = noise(cur_pos, cur_angle) # Зашумление
    if (global_i == len(points)):
         global_i = 1
    target = points[global_i]
    dx = target[0] - cur_pos[0]
    dy = target[2] - cur_pos[2]
    norm = 1/math.sqrt(dx*dx+dy*dy)
    target_vec = [dx*norm, dy*norm]
    gamma = math.acos(target_vec[0]) # Арккос между вектором {1;0} и вектором направления
    if (dy > 0): # Если направление в сторону гамма > Пи
        gamma *= -1
        gamma += math.pi*2
    ro = math.sqrt(dx*dx+dy*dy)
    alfa = gamma - cur_angle
    if (math.fabs(alfa) > math.pi):
        if (cur_angle > gamma):
            alfa = gamma - (cur_angle - math.pi*2)
        if (gamma > cur_angle):
            alfa = alfa - math.pi*2
    k1 = 0.22 # линейная скорость
    k2 = 2.5 # угловая скорость
    k3 = 20 # (не)сила торможения перед КТ
    k4 = 0.07 # радиус КТ
    v = k1*math.tanh(k3*ro)*math.cos(alfa)
    omega = k1*math.tanh(ro)*math.cos(alfa)*math.sin(alfa)/ro+k2*alfa
    print(cur_pos, target)
    print(global_i, "\t", ro, "\t", alfa)
    if (ro < k4):
        print("Came to the target\n")
        global_i += 1
    return([v, omega])
   


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])
    pos = [0.0,0.0,0.0]
    angle = 0
    pos, angle = delay.delay(env.cur_pos, env.cur_angle)
    action = move_to(pos, angle)

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    #angle = env.cur_angle
    #if (angle <0):
    #    angle += 2*math.pi
    #print(angle)
    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    #if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
       # delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
       # v1 += delta_v
       # v2 -= delta_v

    action[0] = v1
    action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    
    #print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    #print(env.cur_pos, "\t",env.cur_angle,"\t", env.speed,"\n")

    if key_handler[key.RETURN]:

        im = Image.fromarray(obs)

        im.save("screen.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
