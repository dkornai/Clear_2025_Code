import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Sec_5_reinforcement_learning.module_lunar import LunarLander

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

### REWARD FUNCTION ###

def reward_fn(state, action):
    '''
    Reward function for the Lunar Lander environment
    Incentivizes the lander to land on the landing pad, with low velocity and upright
    '''

    reward = 0.0    
    
    distance_from_pad = np.sqrt(state[0]**2 + state[1]**2)

    # Positive Reward for the lander being close to the landing pad
    reward -(((-4 * np.clip(    state[1],     0,     1) + 5) * state[0])**2) + 1
    reward += 3*np.exp(-2*state[1])
    
    # Reward for the lander's velocity being close to zero, near the landing pad
    total_velocity = np.sqrt(state[2]**2 + state[3]**2)
    velocity_scaling_factor = np.exp(-distance_from_pad*4 + 0.7)
    reward -= 2*total_velocity*velocity_scaling_factor

    # # Reward for being upright, and having low angular velocity
    angle_scaling_factor =  np.exp(-distance_from_pad)
    reward -= 3*angle_scaling_factor*np.sqrt(state[4]**2)
    
    # Reward for not using engines
    if action == 2:
        reward -= 0.030
    elif action == 1 or action == 3:
        reward -= 0.003

    if abs(state[0]) >= 1.0:
        reward = -100.0
    elif state[1] <= 0.05:
        # If the acceleration suggests that the lander will crash in the next frame, then the reward is -10
        if state[1] + state[3]/50 < -0.05:
            reward = -10.0
        # If the lander has gone through the landing pad, then the reward is -10
        elif state[1] <= -0.03:
            reward = -10.0
        # If the lander is close to the ground and has a low velocity in both the x and y directions, we consider it landed, and the reward is 10
        elif state[1] <= 0.01 and np.abs(state[2]) <= 0.0001 and np.abs(state[3]) <= 0.0001 and np.abs(state[0]) <= 0.1:
            reward = 10.0

    return reward

def reward_fn_torch(state, action_onehot):
    '''
    Reward function rewritten for PyTorch
    '''
    distance_from_pad = torch.sqrt(state[0]**2 + state[1]**2)

    # Compute the initial rewards based on lander's state
    reward = -(((-4 * torch.clamp(state[1], min=0, max=1) + 5) * state[0])**2) + 1
    reward += 3*torch.exp(-2 * state[1])

    # Compute velocity-based rewards
    total_velocity = torch.sqrt(state[2]**2 + state[3]**2)
    velocity_scaling_factor = torch.exp(-distance_from_pad * 4 + 0.7)
    reward -= 2 * total_velocity * velocity_scaling_factor

    # Compute angle-based rewards
    angle_scaling_factor = torch.exp(-distance_from_pad)
    reward -= 3 * angle_scaling_factor * torch.sqrt(state[4]**2)

    # Engine use penalties
    reward -= 0.03 * action_onehot[2]
    reward -= 0.003 * (action_onehot[1] + action_onehot[3])

    # Check extreme conditions and adjust rewards accordingly
    extreme_conditions = (torch.abs(state[0]) >= 1.0)
    reward = torch.where(extreme_conditions & (state[1] + state[3] / 50 < -0.05), torch.tensor(-10.0, device=state.device), reward)
    reward = torch.where(extreme_conditions & (state[1] <= -0.03), torch.tensor(-10.0, device=state.device), reward)
    reward = torch.where(extreme_conditions & (state[1] <= 0.01) & (torch.abs(state[2]) <= 0.0001) & (torch.abs(state[3]) <= 0.0001) & (torch.abs(state[0]) <= 0.1), torch.tensor(10.0, device=state.device), reward)

    return reward