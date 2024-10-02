import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def heuristic_policy(s):
    """
    The heuristic reference policy supplied in the original Lunar Lander environment
    https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py

    Args:
        env: The environment
        s (list): The state. Attributes:
            s[0] is the horizontal coordinate
            s[1] is the vertical coordinate
            s[2] is the horizontal speed
            s[3] is the vertical speed
            s[4] is the angle
            s[5] is the angular speed

    Returns:
         a: action to take
            a=0 is no action
            a=1 is fire left engine
            a=2 is fire main engine
            a=3 is fire right engine
    """

    angle_targ = s[0] * 0.5 + s[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        s[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > +0.05:
        a = 1
    
    return a


class NNPolicy(nn.Module):
    """
    Neural network policy for the Lunar Lander environment
    state and action space are identical to the heuristic reference policy    
    """
    def __init__(self):
        super(NNPolicy, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc5(x)
        return x
    
    def act_train(self, state):
        """
        maps the state vector to an action
        """
        probabilities = torch.nn.functional.softmax(self.forward(state), dim=-1)
        distribution = torch.distributions.Categorical(probabilities)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action_one_hot = F.one_hot(action, num_classes=4).float()
        # returns both the action and the log_prob of the actions
        return action_one_hot, log_prob
        
    def act_eval(self, state):
        """
        maps the state vector to an action
        """
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(self.forward(state), dim=-1)
            probabilities = probabilities.squeeze(1)
            action_indices = probabilities.multinomial(num_samples=1)
            action_one_hot = F.one_hot(action_indices, num_classes=4).float()
        # returns only the action
        return action_one_hot
    

