from __future__ import annotations
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from REINFORCE import REINFORCE
from REINFORCE_SAG import REINFORCE_SAG
from REINFORCE_SGD import REINFORCE_SGD
from REINFORCE_SVRG import REINFORCE_SVRG
import gym
import pandas as pd
import seaborn as sns


plt.rcParams["figure.figsize"] = (10, 5)



# Create and wrap the environment
env = gym.make("InvertedPendulum-v4")
wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]

seed = 2024
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Reinitialize agent every seed
optim = 'SVRG' # [base, SAG, SGD, SVRG]
if optim == 'base':
    agent = REINFORCE(obs_space_dims, action_space_dims)
elif optim == 'SAG':
    agent = REINFORCE_SAG(obs_space_dims, action_space_dims)
elif optim == 'SGD':
    agent = REINFORCE_SGD(obs_space_dims, action_space_dims)
elif optim == 'SVRG':
    agent = REINFORCE_SVRG(obs_space_dims, action_space_dims)
reward_over_episodes = []

for episode in tqdm(range(total_num_episodes)):
    # gymnasium v26 requires users to set seed while resetting the environment
    obs, info = wrapped_env.reset(seed=seed)
    done = False
    episode_reward = 0
    while not done:
        action = agent.sample_action(obs)
        # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
        # These represent the next observation, the reward from the step,
        # if the episode is terminated, if the episode is truncated and
        # additional info from the step
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        agent.rewards.append(reward)
        episode_reward += reward
        # End the episode when either truncated or terminated is true
        #  - truncated: The episode duration reaches max number of timesteps
        #  - terminated: Any of the state space values is no longer finite.
        done = terminated or truncated
    reward_over_episodes.append(episode_reward)
    agent.update()
    if episode % 1000 == 0:
        avg_reward = int(np.mean(wrapped_env.return_queue))
        print("Episode:", episode, "Average Reward:", avg_reward)


rewards_to_plot = [[reward for reward in reward_over_episodes]]
df1 = pd.DataFrame(rewards_to_plot).melt(var_name='episode', value_name='reward')
sns.set(style="darkgrid", context="talk", palette="rainbow")
title = f"REINFORCE_{optim} for InvertedPendulum-v4"
sns.lineplot(x="episode", y="reward", data=df1).set(
    title=title, xlabel="Episode", ylabel="Reward"
)
plt.savefig(f"./{'_'.join(title.split(' '))}.png")
plt.show()
