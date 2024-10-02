import numpy as np
import matplotlib.pyplot as plt

from module_reward import reward_fn

def black_to_null_alpha(img_arr):
    """
    generate an alpha mask of the same size as the original image that has 0 alpha for black pixels
    used in the plotting of trajectories onto a single image
    """
    alpha_mask = (img_arr[:,:,0] != 0) & (img_arr[:,:,1] != 0) & (img_arr[:,:,2] != 0)
    
    return np.array(alpha_mask, dtype=float)

def inputmap(input:int) -> np.ndarray:
    """
    map an action integer [0, 1, 2, 3] to a one-hot encoded array
    """
    out = np.zeros(4)
    out[input] = 1
    return out

def simulate_with_policy(env, repeats, policy_model, plot=False):
    """
    Simulate trajectories in the Lunar Lander environment using a given policy
    
    Parameters:
    env:
        an initialized  LunarLander environment to simulate in
    repeats: int
        the number of trajectories to simulate
    policy_model:
        a function that maps states to actions
    plot: bool
        whether to plot the trajectory of the lander or not

    Returns:
    data_dict: dict
        a dictionary containing the state, input and reward sequences for each trajectory

    """
    if plot:
        assert repeats == 1, 'Plotting only works for a single simulation'

    
    TIMESTEPS = 200
    data_dict = {'STATE':[], 'INPUT':[], 'REWARD':[]}

    # Simulate "repeats" number of trajectories
    for i in range(repeats):
        print(f'Replicate simulation {i+1}/{repeats}', end='\r')

        # Set up the data structures to store the state, input and reward sequences
        state_seq  = np.zeros((TIMESTEPS+2, 8))
        input_seq  = np.zeros((TIMESTEPS+2, 4))
        reward_seq    = np.zeros((TIMESTEPS+2, 1))

        # Reset the environment and get the initial state
        state, _ = env.reset()
        state_seq[0] = state

        # Plot the initial state of the environment if plot is True
        if plot:
            img = env.render()
            plt.imshow(img, alpha=1)
        

        # Simulate the trajectory for TIMESTEPS
        
        for t in range(0, TIMESTEPS):
            action_index = policy_model(state)

            # Input occurs at time t
            input_seq[t] = inputmap(action_index)
            
            # Iterate the environment by one time step
            state, _, _, _, _ = env.step(action_index)
            
            # Get the reward using the reward function
            est_reward = reward_fn(state, action_index)

            # Response to the input is observed at time t+1
            state_seq[t+1]  = state
            reward_seq[t+1] = est_reward
            
            # Plot the current state of the environment if plot is True
            if plot:
                if t % 10 == 0:
                    img = env.render()
                    alpha_mask = np.array(black_to_null_alpha(img), dtype=int)*254
                    rgba_image = np.zeros((400, 600, 4), dtype=int)
                    rgba_image[:,:,:3] = img
                    rgba_image[:,:,3] = alpha_mask
                    plt.imshow(rgba_image)

        # Crop off the first and last time steps
        state_seq  = state_seq[1:-1,0:6] # leg contact is not used
        input_seq  = input_seq[1:-1]
        reward_seq    = reward_seq[1:-1]

        # Store the state, input and reward sequences for the current trajectory in the data_dict
        data_dict['STATE'].append(state_seq)
        data_dict['INPUT'].append(input_seq)
        data_dict['REWARD'].append(reward_seq)
        
        # Plot the reward and state sequences if plot is True
        if plot:
            plt.show()
            plt.figure(figsize=(12, 4))
            plt.plot(reward_seq, label='estimated', linewidth=2)
            plt.title('Reward')
            plt.legend()
            plt.show()

            plt.figure(figsize=(12, 4))
            plt.plot(np.cumsum(reward_seq), label='estimated', linewidth=2) 
            plt.title('Cumulative Reward')
            plt.legend()
            plt.show()

            title = ['pos x', 'pos y', 'vel x', 'vel y', 'angle', 'angular vel']#, 'leg 1', 'leg 2']
            for i in range(6):
                plt.figure(figsize=(12, 2))
                plt.plot(state_seq[:,i], linewidth=2)
                plt.title(title[i])
                plt.show()

    print()

    return data_dict