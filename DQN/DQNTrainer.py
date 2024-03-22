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
    def __init__(self, 
                 env_name, 
                 in_channels=1, 
                 learning_rate=1e-4, 
                 buffer_size=10000, 
                 epsilon = 0.9, 
                 gamma = 0.95,
                 log_level = 1,
                 ):
        # 获取当前时间
        current_time = datetime.now()

        # 将日期时间对象转换为特定格式的字符串
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.writer = SummaryWriter(log_dir=f'logs/DQN/{formatted_time}-env:{env_name}-lr:{learning_rate}-bs:{buffer_size}-eps:{epsilon}-gamma:{gamma}')
        self.env = Env.DQNenv(env_name)
        self.n_actions = self.env.n_actions
        self.env.info()
        self.agent = DQNAgent.DQN_agent(in_channels=in_channels,
                                        n_actions=self.n_actions,
                                        learning_rate=learning_rate,
                                        buffer_size=buffer_size, 
                                        epsilon = epsilon,
                                        gamma = gamma,
                                        log_level=log_level,
                                        )
        self.rewards = []
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
        
    def train(self, max_frame=1000):
        print("Start training...")
        state = self.env.env.reset()
        state = state[0]
        terminated = False
        truncated = False
        episode_reward = 0
        episode = 0
        for frame in range(max_frame):
            print(f'frame: {frame}')
            if terminated or truncated:
                state = self.env.env.reset()
                # state = self.get_state(state[0])
                state = state[0]
                self.writer.add_scalar('episode reward', episode_reward, frame)
                episode+=1
                episode_reward = 0
            actions = self.agent.get_action(state)
            action = actions[0]
            observation, reward, terminated, truncated, info = self.env.env.step(action)
            if frame % 4 == 0:
                self.agent.receive_response(state, reward, action, observation)
            state = observation
            episode_reward += reward
            self.rewards.append(reward)
            # print(f'frame: {frame}, reward: {reward}')
            self.writer.add_scalar('reward', reward, frame)
            loss = self.agent.train()
            self.losses.append(loss)
            self.writer.add_scalar('loss', loss, frame)
            # self.writer.add_scalar('i',i,episode)
            # print(f'frame: {frame}, loss: {loss}')
            if frame % 100 == 0:
                torch.cuda.empty_cache()
            
    def paint(self):
        plt.plot(self.rewards)
        plt.savefig('./image/rewards.png')
        plt.plot(self.losses)
        plt.savefig('./image/losses.png')
            
    def test(self):
        self.env.testEnv()
    