# tools
import torch
from torch import nn
from torchvision import transforms as T
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy
import time, datetime, copy
# SWA
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
# gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# import network
from network import Qnet
# stage
from tools import make_stage, Replaybuffer

class MarioAgent():
    def __init__(self, c):
        # all stages that are used for eval
        stage1, stage2, stage3, stage4 = make_stage() 
        self.stage1 = stage1
        self.stage2 = stage2
        self.stage3 = stage3
        self.stage4 = stage4
        # hyper prametor
        self.gamma = 0.98
        self.lr = 0.00005
        self.buffer_size = 1000
        self.batch_size = 256
        self.action_size = len(SIMPLE_MOVEMENT)
        self.frame_size = (4, 84, 84)
        # Averaged cycle
        self.c = c 
        # Epsilon for exploit and explore
        self.epsilon = 1
        self.epsilon_decay = 0.99978
        self.epsilon_min = 0.1
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Replay Buffer
        self.replay_buffer = Replaybuffer(self.buffer_size, self.batch_size)
        # network
        self.qnet = Qnet(self.frame_size, self.action_size).to(self.device)
        self.target_qnet = Qnet(self.frame_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr = self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        # swa network
        self.swa_qnet = torch.optim.swa_utils.AveragedModel(self.qnet)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.lr)

    def update_target_qnet(self):
        self.target_qnet = copy.deepcopy(self.qnet)        

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(len(SIMPLE_MOVEMENT))
        else:
            state = state.__array__() 
            state = torch.tensor(state).cuda()
            state = state.unsqueeze(0)
            qs = self.qnet(state)
            action_idx = torch.argmax(qs, axis=1).item()     
        return action_idx
    
    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        return self.epsilon

    def update(self, state, action, reward, next_state, done, swa_start, episode):
        self.replay_buffer.add(state.__array__(), action, reward, next_state.__array__(), done)
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state) # qs : (256, 7), state : (256, 4*84*84)
        q = qs[np.arange(self.batch_size), action] 

        next_qs = self.target_qnet(next_state)
        next_qs = next_qs.to('cpu').detach().numpy().copy()
        next_q = next_qs.max(axis=1)
        b_done = []
        for i in done:
            if i:
                b_done.append(1.0)
            else:
                b_done.append(0.0)

        reward = reward.to('cpu').detach().numpy().copy()
        target = []

        for i in range(len(reward)):
            t = reward[i] + (1.0 - b_done[i])*self.gamma*next_q.max()
            target.append(t)

        target = torch.tensor(target).cuda()
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        if episode >= swa_start:
            if episode % self.c == 0:
                self.swa_qnet.update_parameters(self.qnet)
        self.swa_scheduler.step()

        return loss        


    def eval_stage(self, stage_num):
        if stage_num == 1:
            env = self.stage1
        elif stage_num == 2:
            env = self.stage2
        elif stage_num == 3:
            env = self.stage3
        elif stage_num == 4:
            env = self.stage4
        else:
            print("We can't find stage number(use stage1)")
            env = self.stage1
        
        done = False
        total_reward = 0
        state = env.reset()
        while not done:
            state = state.__array__()
            state = torch.tensor(state).cuda()
            state = state.unsqueeze(0)
            qs = self.qnet(state)
            action = torch.argmax(qs, axis=1).item()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
        
        return total_reward
    
    def eval_swa(self, stage_num):
        if stage_num == 1:
            env = self.stage1
        elif stage_num == 2:
            env = self.stage2
        elif stage_num == 3:
            env = self.stage3
        elif stage_num == 4:
            env = self.stage4
        else:
            print("We can't find stage number(use stage1)")
            env = self.stage1


        done = False
        total_reward = 0
        state = env.reset()
        while not done:
            state = state.__array__()
            state = torch.tensor(state).cuda()
            state = state.unsqueeze(0)
            qs = self.swa_qnet(state)
            action = torch.argmax(qs, axis=1).item()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state    
        
        return total_reward
