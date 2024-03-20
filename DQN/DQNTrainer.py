import DQN.DQNAgent as DQNAgent
import utility.EnvConfig as Env
import matplotlib.pyplot as plt
import torch
import numpy as np

class DQNTrainer(object):
    def __init__(self, env_name, in_channels=1, learning_rate=1e-4, buffer_size=10000, epsilon = 0.95, gamma = 0.95):
        self.env = Env.DQNenv(env_name)
        self.n_actions = self.env.n_actions
        self.env.info()
        self.agent = DQNAgent.DQN_agent(in_channels=in_channels, n_actions=self.n_actions, learning_rate=learning_rate, buffer_size=buffer_size, epsilon = epsilon, gamma = gamma)
        self.rewards = []
        self.losses = []
        
    def train(self, max_episode=1000):
        print("Start training...")
        for episode in range(max_episode):
            print(f'episode: {episode}')
            state = self.env.env.reset()
            # print(f'state:\n {state[0].shape}')
            state = torch.tensor(state[0], dtype=torch.float32)
            state = state.unsqueeze(0)
            # print(f'state in trainer:\n {state.shape}')
            episode_reward = 0
            i = 0
            while True:
                print(f"i: {i}")
                i += 1
                assert(state.shape == (1,84, 84))
                action = self.agent.get_action(state)
                observation, reward, terminated, truncated, info = self.env.env.step(action)
                # next_state, reward, done, _ = self.env.env.step(action)
                observation = torch.tensor(observation, dtype=torch.float32)
                observation = observation.unsqueeze(0)
                self.agent.receive_response(state, reward, action, observation)
                episode_reward += reward
                state = observation
                if terminated or truncated or i >= 1000:
                    break
            self.rewards.append(episode_reward)
            print(f'episode: {episode}, reward: {episode_reward}')
            loss = self.agent.train()
            self.losses.append(loss)
            print(f'episode: {episode}, loss: {loss}')
            
    def paint(self):
        plt.plot(self.rewards)
        plt.savefig('./image/rewards.png')
        plt.plot(self.losses)
        plt.savefig('./image/losses.png')
            
    def test(self):
        self.env.testEnv()
    