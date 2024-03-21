import DQN.DQNAgent as DQNAgent
import utility.EnvConfig as Env
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter   

class DQNTrainer(object):
    def __init__(self, env_name, in_channels=1, learning_rate=1e-4, buffer_size=10000, epsilon = 0.9, gamma = 0.95):
        # 获取当前时间
        current_time = datetime.now()

        # 将日期时间对象转换为特定格式的字符串
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.writer = SummaryWriter(log_dir=f'logs/DQN/{formatted_time}')
        self.env = Env.DQNenv(env_name)
        self.n_actions = self.env.n_actions
        self.env.info()
        self.agent = DQNAgent.DQN_agent(in_channels=in_channels, n_actions=self.n_actions, learning_rate=learning_rate, buffer_size=buffer_size, epsilon = epsilon, gamma = gamma)
        self.rewards = []
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
        
    def get_state(self,observation):
        observation = list(observation)
        state = np.array(observation, dtype=np.float32)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = state.to(self.device)
        return state

    def train(self, max_episode=1000):
        print("Start training...")
        for episode in range(max_episode):
            print(f'episode: {episode}')
            state = self.env.env.reset()
            # state = self.get_state(state[0])
            state = state[0]
            episode_reward = 0
            i = 0
            while True:
                i += 1
                # assert(state.shape == (1,84,84) and state.dtype == torch.float32)
                actions = self.agent.get_action(state)
                action = actions[0]
                observation, reward, terminated, truncated, info = self.env.env.step(action)
                # observation = self.get_state(observation)
                self.agent.receive_response(state, reward, action, observation)
                episode_reward += reward
                state = observation
                # self.writer.add_scalar(f'reward {episode}', reward, i)
                if terminated or truncated:
                    break
            self.rewards.append(episode_reward)
            print(f'episode: {episode}, reward: {episode_reward}')
            self.writer.add_scalar('reward', episode_reward, episode)
            loss = self.agent.train()
            self.losses.append(loss)
            self.writer.add_scalar('loss', loss, episode)
            # self.writer.add_scalar('i',i,episode)
            print(f'episode: {episode}, loss: {loss}')
            
    def paint(self):
        plt.plot(self.rewards)
        plt.savefig('./image/rewards.png')
        plt.plot(self.losses)
        plt.savefig('./image/losses.png')
            
    def test(self):
        self.env.testEnv()
    