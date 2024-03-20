import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
import numpy as np
import matplotlib.pyplot as plt

class DQNenv(object):
    def __init__(self,name) -> None:
        self.name = name
        self.env = GrayScaleObservation(gym.make(name))
        self.env = ResizeObservation(self.env, 84)
        self.n_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape

    def info(self):
        print(f'env_name: {self.name}, n_actions: {self.n_actions}, state_dim: {self.state_dim}')
        
    def testEnv(self):
        rewards = []    
        observation, info = self.env.reset(seed=42)
        for _ in range(1000):
            action = self.env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)  
            
            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()
        plt.plot(rewards)
        plt.imshow(observation)
        print(f'action: {action} info: {info},w:{observation.shape[1]},h:{observation.shape[0]}')
        # print(state)
        plt.savefig('./image/testEnv.png')
        self.env.close()
        
if __name__ == "__main__":
    Env = DQNenv('PongNoFrameskip-v4')
    Env.info()