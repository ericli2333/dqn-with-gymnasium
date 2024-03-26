import DQN.DQNAgent as DQNAgent
import utility.EnvConfig as Env
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys
import math
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter   

class DQNTrainer(object):
    def __init__(self, 
                 env_name, 
                 in_channels=4, 
                 learning_rate=1e-4, 
                 buffer_capacity=10000, 
                 epsilon = 0.9, 
                 epsilon_lower_bound = 0.03,
                 epsilon_upper_bound = 0.9,
                 eps_decay = 200000,
                 gamma = 0.99,
                 log_level = 1,
                 ):
        self.epsilon_lower_bound = epsilon_lower_bound
        self.epsilon_upper_bound = epsilon_upper_bound
        self.eps_decay = eps_decay
        # 获取当前时间
        current_time = datetime.now()

        # 将日期时间对象转换为特定格式的字符串
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_level = log_level
        if log_level == 1:
            self.writer = SummaryWriter(log_dir=f'logs/DQN/{formatted_time}-env:{env_name}-lr:{learning_rate}-bs:{buffer_capacity}-eps:{epsilon}-gamma:{gamma}')
        self.env = Env.DQNenv(env_name)
        self.n_actions = self.env.n_actions
        self.env.info()
        self.agent = DQNAgent.DQN_agent(
                                        writer=self.writer,
                                        in_channels=in_channels,
                                        n_actions=self.n_actions,
                                        learning_rate=learning_rate,
                                        buffer_capacity=buffer_capacity, 
                                        epsilon = epsilon,
                                        gamma = gamma,
                                        log_level=log_level,
                                        )
        self.rewards = []
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
        
    def calculate_epsilon(self, frame_idx):
        return self.epsilon_lower_bound + (self.epsilon_upper_bound - self.epsilon_lower_bound) * math.exp(
            -1. * frame_idx / self.eps_decay)

    def train(self, max_frame=1000):
        print("Start training...")
        state = self.env.env.reset()
        state = state[0]
        action_list = []
        terminated = True
        truncated = True
        episode_reward = 0
        episode = 0
        for frame in range(max_frame):
            eps = self.calculate_epsilon(frame)
            print(f'frame: {frame}')
            if terminated or truncated:
                state = self.env.env.reset()
                eps = 0
                # state = self.get_state(state[0])
                state = state[0]
                if self.log_level == 1:
                    self.writer.add_scalar('episode reward', episode_reward, frame)
                episode += 1
                episode_reward = 0
            actions = self.agent.get_action(state, eps)
            assert(type(actions) == int)
            action_list.append(actions)
            # action = actions
            observation, reward, terminated, truncated, info = self.env.env.step(actions)
            self.agent.receive_response(state, reward, actions, observation, terminated or truncated)
            state = observation
            episode_reward += reward
            self.rewards.append(reward)
            loss = self.agent.train(frame)
            self.losses.append(loss)
            if self.log_level == 1:
                # self.writer.add_scalar('reward', reward, frame)
                self.writer.add_scalar('epsilon',eps, frame)
                self.writer.add_scalar('batch current size',self.agent.replay_buffer.curSize,frame)
                self.writer.add_scalar('loss', loss, frame)
                if frame % 1000 == 0:
                    self.writer.add_histogram('action', np.array(action_list,dtype=int), frame // 1000)
                    action_list = []
            if self.log_level == 2:
                print(f'frame: {frame}, reward: {reward}')
                print(f'frame: {frame}, loss: {loss}')
            if frame % 100 == 0:
                torch.cuda.empty_cache()
            
    def paint(self):
        plt.plot(self.rewards)
        plt.savefig('./image/rewards.png')
        plt.plot(self.losses)
        plt.savefig('./image/losses.png')
            
    def test(self):
        self.env.testEnv()
    