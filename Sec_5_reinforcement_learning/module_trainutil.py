import numpy as np
import matplotlib.pyplot as plt

### STATE SELECTION FOR TRAINING DATA ###
def state_selector(data_dict, max_t_sample, sample_size):
    """
    randomly select states from the data_dict
    """

    state_list = []
    selected = 0
    
    while selected < sample_size:
        random_rep_index = np.random.randint(0, len(data_dict['STATE']))
        
        # Select a random starting time index up to max_t_sample
        random_time_index = np.random.randint(1, max_t_sample)
        
        ## count the number of times the maximum reward is achieved in the next 10 time steps
        #times_max_reward_achieved = np.sum(data_dict['REWARD'][random_rep_index][:random_time_index+10] >= 9)
        #landed = times_max_reward_achieved > 5
        # or check if the craft rotated more than 0.5 radians
        maximum_angle  = np.max(data_dict['STATE'][random_rep_index][:random_time_index,4])
        over_rotated = maximum_angle >= 0.5
        # or crashed (reward = -10)
        minimum_reward = np.min(data_dict['REWARD'][random_rep_index][:random_time_index])
        crashed = minimum_reward < -9
        # or went outside x = 0.5
        maximum_x      = np.max(np.abs(data_dict['STATE'][random_rep_index][:random_time_index,0]))
        outside_x = maximum_x >= 0.5
        
        # do not include the state in the training set
        if not crashed and not over_rotated and not outside_x:# and not landed:
            state = data_dict['STATE'][random_rep_index][random_time_index]
            state_list.append(np.array([state]))
            selected += 1
    
    return state_list

def crash_challenge_state_selector(data_dict, max_t_sample, sample_size):
    """
    Find challenging states where the first crash occurs in the next HORIZON time steps
    """

    state_list = []
    selected = 0

    crashes = extract_crashes(data_dict)
    
    options = []
    for crash_rep_index, crash_time in crashes:
        # if the lander rotated more than 0.4 radians
        maximum_angle  = np.max(data_dict['STATE'][crash_rep_index][:,4])
        over_rotated = maximum_angle >= 0.4
        # or went outside x = 0.11 up to the crash time
        maximum_x      = np.max(np.abs(data_dict['STATE'][crash_rep_index][:crash_time,0]))
        outside_x = maximum_x >= 0.11

        # do not include the state in the challenge set
        if not over_rotated and not outside_x:
            options.append([crash_rep_index, crash_time])

    if len(options) < sample_size:
        return []
    
    while selected < sample_size:
        index = np.random.randint(0, len(options))
        rep_index, time_index = options[index]
        time_index = np.random.randint(time_index-15, time_index-5)
        state = data_dict['STATE'][rep_index][time_index]
        state_list.append(np.array([state]))
        selected += 1
    
    return state_list

def x_challenge_state_selector(data_dict, max_t_sample, sample_size):
    """
    Find challenging but solveable states where the lander is within 0.05 <= x <= 0.2 but above y = 0.5
    """

    state_list = []

    states = np.array(data_dict['STATE']) # (repeats, timesteps, 6)
    xy_states = states[:,:,0:2] # (repeats, timesteps, 2)
    x_states = np.abs(xy_states[:,:,0]) # (repeats, timesteps)
    y_states = xy_states[:,:,1] # (repeats, timesteps)

    options = []

    for i in range(len(data_dict['STATE'])):
        for t in range(1, max_t_sample):
            if y_states[i][t] >= 0.5 and np.abs(x_states[i][t]) >= 0.05 and np.abs(x_states[i][t]) <= 0.2:
                state = data_dict['STATE'][i][t]
                options.append(np.array([state]))

    if len(options) < sample_size:
        return []
    
    selected_indices = np.random.choice(len(options), sample_size, replace=False)
    for index in selected_indices:
        state_list.append(options[index])

    return state_list

### PERFORMANCE METRICS ###
def extract_rewards_over_time(data_dict):
    """
    Extract the mean and std for the momentary and cumulative rewards over time
    """
    reward = np.array(data_dict['REWARD']) # (repeats, timesteps, 1)
    reward_mean = np.mean(reward, axis=0)  # (timesteps, 1)
    reward_std  = np.std(reward, axis=0)   # (timesteps, 1)
    c_reward = np.cumsum(reward, axis=1)      # (repeats, timesteps, 1)
    c_reward_mean = np.mean(c_reward, axis=0) # (timesteps, 1)
    c_reward_std  = np.std(c_reward, axis=0)  # (timesteps, 1)
    
    reward_t_mean = np.round(reward_mean.reshape(-1), 2)
    reward_t_std  = np.round(reward_std.reshape(-1), 2)
    reward_ct_mean = np.round(c_reward_mean.reshape(-1), 2)
    reward_ct_std  = np.round(c_reward_std.reshape(-1), 2)

    return reward_t_mean, reward_t_std, reward_ct_mean, reward_ct_std

def extract_final_rewards(data_dict):
    reward_t_mean, reward_t_std, reward_ct_mean, reward_ct_std = extract_rewards_over_time(data_dict)
    return reward_t_mean[-1], reward_t_std[-1], reward_ct_mean[-1], reward_ct_std[-1]

def extract_crashes(data_dict):
    """
    Get pairs of (replicate, timestep) where the lander crashed for the first time
    """
    crashed = []
    rewards = data_dict['REWARD'] #[(timesteps, 1)]
    rewards = [reward.reshape(-1).tolist() for reward in rewards]
    for i, reward in enumerate(rewards):
        for t, r in enumerate(reward):
            if r < -9:
                crashed.append([i, t])
                break

    return crashed

def extract_crash_land_stats(data_dict):
    rewards = data_dict['REWARD'] #[(timesteps, 1)]
    rewards = [reward.reshape(-1).tolist() for reward in rewards]

    crashed = 0
    landed = 0
    for reward in rewards:
        if any(r < -9 for r in reward):
            crashed += 1
        elif any(r > 9 for r in reward):
            landed += 1

    crashed_percentage = np.round((crashed/len(rewards))*100 , 2)
    landed_percentage = np.round((landed/len(rewards))*100 , 2)

    return crashed_percentage, landed_percentage

def extract_land_timings(data_dict):
    rewards = data_dict['REWARD'] #[(timesteps, 1)]
    rewards = [reward.reshape(-1).tolist() for reward in rewards]

    landing_times = []
    for reward in rewards:
        for t, r in enumerate(reward):
            if r > 9:
                landing_times.append(t)
                break

    mean_landing_time = np.round(np.mean(landing_times), 2)
    std_landing_time  = np.round(np.std(landing_times), 2)

    return mean_landing_time, std_landing_time

def extract_fuel_consumption(data_dict):
    """
    Get the mean and std of total fuel consumption across a set of simulations
    """
    fuel_consumption = []
    for r in range (len(data_dict['INPUT'])):
        inputs = data_dict['INPUT'][r] #(timesteps, 4)
        left_engine  = np.sum(inputs[:,1])*0.03
        main_engine  = np.sum(inputs[:,2])*0.3
        right_engine = np.sum(inputs[:,3])*0.03
        fuel_consumption.append(left_engine + main_engine + right_engine)

    mean_fuel_consumption = np.round(np.mean(fuel_consumption), 2)
    std_fuel_consumption  = np.round(np.std(fuel_consumption), 2)

    return mean_fuel_consumption, std_fuel_consumption
        
def update_results_list(result_list, result_name, replicate_index, data_dict):
    
    final_reward_mean, final_reward_std, final_c_reward_mean, final_c_reward_std = extract_final_rewards(data_dict)
    crashed_percentage, landed_percentage = extract_crash_land_stats(data_dict)
    mean_landing_time, std_landing_time = extract_land_timings(data_dict)
    mean_fuel_consumption, std_fuel_consumption = extract_fuel_consumption(data_dict)

    print(f'\n{result_name} (rep. {replicate_index})\nFinal Reward: {final_c_reward_mean}(+-{final_c_reward_std}), C: {crashed_percentage}% L: {landed_percentage}%, Lt: {mean_landing_time}(+-{std_landing_time}), Fuel: {mean_fuel_consumption}(+-{std_fuel_consumption})\n')

    result_list.append({
        'rep_i':replicate_index, 
        'result_name':result_name, 
        'c_reward':final_c_reward_mean, 
        'crashed':crashed_percentage, 
        'landed':landed_percentage, 
        'mean_landing_time':mean_landing_time, 
        'mean_fuel_consumption':mean_fuel_consumption,
        })
    
    return result_list

def update_rewards_over_time_list(result_list, result_name, data_dict):
    
    reward_t_mean, reward_t_std, reward_ct_mean, reward_ct_std = extract_rewards_over_time(data_dict)
    
    result_list.append({
        'result_name':result_name, 
        'reward_t_mean':reward_t_mean, 
        'reward_t_std':reward_t_std, 
        'reward_ct_mean':reward_ct_mean, 
        'reward_ct_std':reward_ct_std
        })
    
    return result_list

def reward_over_time_plotting(rewards_over_time_list):
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].set_title('Mean Reward at Time')
    axs[1].set_title('Mean Cumulative Reward')
    for result in rewards_over_time_list:
        label = result['result_name']
        reward_mean = result['reward_t_mean']
        c_reward_mean = result['reward_ct_mean']
    
        axs[0].plot(reward_mean, label=label)
        axs[1].plot(c_reward_mean, label=label)
    
    axs[1].legend(loc = 'upper left')
    plt.show()
