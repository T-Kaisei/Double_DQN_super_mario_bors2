import random
import numpy as np
from collections import deque
# pytorch
import torch
from torchvision import transforms as T
#openAI
import gym
from gym.spaces import Box
# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace
# Mario Enviroment
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# Edit picture
from gym.wrappers import RecordVideo
from gym.wrappers import FrameStack


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class Replaybuffer(object):
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        state = state.__array__()
        state = torch.tensor(state).cuda()
        next_state = next_state.__array__()
        next_state = torch.tensor(next_state).cuda()
        action = torch.tensor([action]).cuda()
        reward = torch.tensor([reward]).cuda()
        done = torch.tensor([done]).cuda()
        self.buffer.append((state, action, reward, next_state, done,))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

# make stage
def make_stage():
    # All enviroment for experiment
    stage1 = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", new_step_api=True)
    stage1 = JoypadSpace(stage1, SIMPLE_MOVEMENT)
    stage1 = SkipFrame(stage1, skip=4)
    stage1 = GrayScaleObservation(stage1)
    stage1 = ResizeObservation(stage1, shape=84)
    stage1 = FrameStack(stage1, num_stack=4)

    stage2 = gym_super_mario_bros.make("SuperMarioBros-1-2-v0", new_step_api=True) 
    stage2 = JoypadSpace(stage2, SIMPLE_MOVEMENT)
    stage2 = SkipFrame(stage2, skip=4)
    stage2 = GrayScaleObservation(stage2)
    stage2 = ResizeObservation(stage2, shape=84)
    stage2 = FrameStack(stage2, num_stack=4)

    stage3= gym_super_mario_bros.make("SuperMarioBros-1-3-v0", new_step_api=True) 
    stage3= JoypadSpace(stage3, SIMPLE_MOVEMENT)
    stage3= SkipFrame(stage3, skip=4)
    stage3= GrayScaleObservation(stage3)
    stage3= ResizeObservation(stage3, shape=84)
    stage3= FrameStack(stage3, num_stack=4)

    stage4 = gym_super_mario_bros.make("SuperMarioBros-1-4-v0", new_step_api=True)
    stage4 = JoypadSpace(stage4, SIMPLE_MOVEMENT)
    stage4 = SkipFrame(stage4, skip=4)
    stage4 = GrayScaleObservation(stage4)
    stage4 = ResizeObservation(stage4, shape=84)
    stage4 = FrameStack(stage4, num_stack=4)
    
    return stage1, stage2, stage3, stage4

# set seed
def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True