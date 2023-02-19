import torch
from torch import nn
# from other files
from network import Qnet
from mario import MarioAgent

import gym
from gym.spaces import Box
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from tools import make_stage

def check_gpu():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('GPU is available')
    else:
        device = torch.device("cpu")
        print('GPU is not available')

def check_mario_gym():
    stage1, stage2, stage3, stage4 = make_stage()
    env = stage2
    state = env.reset()  # reset for each new trial
    done = False
    while not done: 
        action = env.action_space.sample()  # select a random action
        next_state, reward, done, info = env.step(action)
        #env.render()
    print("pass")


if __name__ == '__main__':
    check_gpu()
    check_mario_gym()
    e = 1
    d = 0.99997
    for i in range(50000):
        e *= d
        if i % 10000 ==0:
            print(e) 
