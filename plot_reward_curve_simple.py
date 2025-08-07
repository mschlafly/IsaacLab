import torch
import matplotlib.pyplot as plt

rewards = []
for agent in ['2400','4800','7200','9600', '12000','14400','16800','19200','21600','24000']:
    file_path = 'logs/skrl/reach_franka/2025-08-07_13-21-29_ppo_torch/checkpoints/agent_'+agent+'.pt'

    try:
        data = torch.load(file_path, map_location='cpu')  # Use 'cuda' if needed

        # # Print keys to understand the structure
        # print("Available keys:", data.keys())

        # Try to extract rewards (adjust the key based on your file)
        if 'value_preprocessor' in data:
            reward = data['value_preprocessor']['running_mean']
        print(reward)
        rewards.append(reward.tolist())
        # elif 'episode_rewards' in data:
        #     rewards = data['episode_rewards']
        # else:
        #     raise KeyError("No recognizable reward key found in the .pt file.")
    except:
        print('file did not load', file_path)
        

print(rewards)


# # Convert to list if it's a tensor
# if isinstance(rewards, torch.Tensor):
#     rewards = rewards.tolist()

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards, label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Progression Over Training')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
