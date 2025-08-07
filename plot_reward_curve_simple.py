import torch
import matplotlib.pyplot as plt

filename = '2025-08-07_17-00-58_ppo_torch'

rewards = []
# for agent in ['2400','4800','7200','9600', '12000','14400','16800','19200','21600','24000']:
for agent in list(range(5000, 50001, 5000)):

    file_path = 'logs/skrl/reach_franka/'+filename+'/checkpoints/agent_'+str(agent)+'.pt'

    try:
        data = torch.load(file_path, map_location='cpu')  # Use 'cuda' if needed

        # # Print keys to understand the structure
        # print("Available keys:", data.keys())
        if 'value_preprocessor' in data:
            reward = data['value_preprocessor']['running_mean']
        # print(reward)
        rewards.append(reward.tolist())
        # elif 'episode_rewards' in data:
        #     rewards = data['episode_rewards']
        # else:
        #     raise KeyError("No recognizable reward key found in the .pt file.")
    except:
        print('file did not load', file_path)
        

# print(rewards)


# # Convert to list if it's a tensor
# if isinstance(rewards, torch.Tensor):
#     rewards = rewards.tolist()

plt.figure(figsize=(10, 5))
plt.plot(rewards, label='Episode Reward')
# plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Progression Over Training')
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

import pickle

def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

i = 0
file_path = f'logs/skrl/reach_franka/2025-08-07_14-39-11_ppo_torch/Checkpoints/data_ep{i}.pkl'
data = load_pickle_data(file_path)

# print(data['env_interaction'])

# print(data['rewards'])

plt.figure(figsize=(10, 5))
plt.plot(data['env_interaction'],data["rewards"], label="rewards")
plt.plot(data['env_interaction'],data["dist_reward"], label="dist_reward")
plt.plot(data['env_interaction'],data["rot_reward"], label="rot_reward")
plt.plot(data['env_interaction'],data["action_penalty"], label="action_penalty")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Progression Over Training pickle file')
plt.grid(True)
plt.legend()
plt.legend()
plt.tight_layout()


import os
import numpy as np

env_interaction = [0]
rewards = [0]
dist_reward = [0]
rot_reward = [0]
action_penalty = [0]
for i in range(1000):
    file_path = 'logs/skrl/reach_franka/'+filename+f'/Checkpoints/mydata/data_ep{i}.pkl'
    print(file_path)
    if os.path.exists(file_path):
        data = load_pickle_data(file_path)
        env_interaction.append(env_interaction[-1] + data['env_interaction'][-1])
        rewards.append(np.sum(data['rewards']))
        dist_reward.append(np.sum(data['dist_reward']))
        rot_reward.append(np.sum(data['rot_reward']))
        action_penalty.append(np.sum(data['action_penalty']))
        # print(np.mean(data['rewards']))

# print(env_interaction)
plt.figure(figsize=(10, 5))
plt.plot(env_interaction, rewards, label="total reward")
plt.plot(env_interaction, dist_reward, label="dist_reward")
plt.plot(env_interaction, rot_reward, label="rot_reward")
plt.plot(env_interaction, action_penalty, label="action_penalty")
plt.xlabel('Environment Interations')
plt.ylabel('Reward')
plt.title('Reward During Training')
plt.grid(True)
plt.legend()
plt.legend()
plt.tight_layout()

plt.show()

