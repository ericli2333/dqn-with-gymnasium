"""
环境配置
"""

import gymnasium as gym
import random, pickle, os.path, math, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from gymnasium.wrappers import AtariPreprocessing, ResizeObservation,LazyFrames, FrameStack,GrayScaleObservation

# from IPython.display import clear_output
from torch.utils.tensorboard.writer import SummaryWriter

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor # type: ignore
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)



"""
DQN网络搭建
"""


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)


"""
经验回收
"""

class Memory_Buffer(object):
    def __init__(self, memory_size=100000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size: # buffer not full
            self.buffer.append(data)
        else: # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):

            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done= data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)


"""
Agent 设置
"""

class DQNAgent:
    def __init__(self, in_channels = 1, action_space = [], USE_CUDA = False, memory_size = 10000, epsilon  = 1, lr = 1e-4):
        self.epsilon = epsilon
        self.action_space = action_space
        self.memory_buffer = Memory_Buffer(memory_size)
        self.DQN = DQN(in_channels = in_channels, num_actions = action_space.n)
        self.DQN_target = DQN(in_channels = in_channels, num_actions = action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())
        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
            """
            if torch.cuda.device_count() > 1: # 含有多张GPU的卡
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # 单机多卡DP训练
                self.DQN = nn.DataParallel(self.DQN)
                self.DQN_target = nn.DataParallel(self.DQN_target) 
            """
        self.optimizer = optim.RMSprop(self.DQN.parameters(),lr=lr, eps=0.001, alpha=0.95)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        # print(*lazyframe[0].__array__())
        # print(lazyframe[0].shape)
        if lazyframe[0].shape == (4,84,84):
            state =  torch.from_numpy(np.array(lazyframe[0].__array__()[None])).float()
        else:
            state = torch.from_numpy(np.array(lazyframe)).float()
        # print(state.shape)
        # input("pause")
        if self.USE_CUDA:
            state = state.cuda()
        # if state.shape == (84,84):
        #     print("reshape")
        #     state = state.unsqueeze(0).repeat(1, 4, 1, 1)
        # print(state)
        # input("Pause")
        return state

    def value(self, state):
        # print(state)
        # assert(state.shape == (4,84,84))
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon = None):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        # print(f'state:{state.shape} {type(state)}')
        # input()
        if epsilon is None: epsilon = self.epsilon

        if random.random()<epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            assert(state.shape == (4,84,84))
            state = state.unsqueeze(0)
            print(state.shape)
            q_values = self.value(state).cpu().detach().numpy()
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        actions = torch.tensor(actions).long()    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype =torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).bool()  # shape: [batch_size]

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q-values for all actions in current states
        # print(f'state:{states.shape}')
        # input()
        predicted_qvalues = self.DQN(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.DQN_target(next_states) # YOUR CODE

        # compute V*(next_states) using predicted next q-values
        next_state_values =  predicted_next_qvalues.max(-1)[0] # YOUR CODE

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + gamma *next_state_values # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        #loss = torch.mean((predicted_qvalues_for_actions -
        #                   target_qvalues_for_actions.detach()) ** 2)
        loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())

        return loss

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done= data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.stack(states), actions, rewards, torch.stack(next_states), dones

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in self.DQN.parameters():
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return(td_loss.item())
        else:
            return(0)


"""
实验参数设置
"""

if __name__ == '__main__':

    # Training DQN in PongNoFrameskip-v4
    env = GrayScaleObservation(gym.make('PongNoFrameskip-v4'))
    env = ResizeObservation(env, 84)
    env = FrameStack(env, num_stack=4)
    # env = AtariPreprocessing(env,
    #                          scale_obs=False,
    #                          terminal_on_life_loss=True,
    #                          )


    gamma = 0.99
    epsilon_max = 1
    epsilon_min = 0.02
    eps_decay = 200000 # 增加了随机性
    frames = 2000000
    USE_CUDA = True
    learning_rate = 2e-4
    max_buff = 100000
    update_tar_interval = 1000
    batch_size = 64 # 增加了经验回收容量
    print_interval = 1000
    log_interval = 1000
    learning_start = 10000
    win_reward = 19    # 增强了奖励要求
    win_break = True

    action_space = env.action_space
    action_dim = env.action_space.n


    """
    Training process
    """

    state_dim = env.observation_space.shape[1]
    state_channel = env.observation_space.shape[0]

    agent = DQNAgent(in_channels = state_channel, action_space= action_space, USE_CUDA = USE_CUDA, lr = learning_rate, memory_size = max_buff)

    frame = env.reset()

    episode_reward = 0
    all_rewards = []
    losses = []
    episode_num = 0
    is_win = False
    # tensorboard
    summary_writer = SummaryWriter(log_dir = "DQN_00", comment= "good_makeatari")

    # e-greedy decay
    epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
                -1. * frame_idx / eps_decay)
    # plt.plot([epsilon_by_frame(i) for i in range(10000)])

    print(agent.USE_CUDA)

    first_frame = True

    for i in range(frames):
        print(i)
        epsilon = epsilon_by_frame(i)
        
        state_tensor = agent.observe(frame)
        if first_frame:
            first_frame = False
            # state_tensor = state_tensor[0]
            state_tensor = state_tensor.squeeze(0)
            epsilon = 1
        action = agent.act(state_tensor, epsilon)

        next_frame, reward, terminated ,truncated , info= env.step(action)
        assert(next_frame.shape == (4,84,84))
        # print(*next_frame.shape)
        # input("Pause...")

        episode_reward += reward
        if not first_frame:
            if agent.observe(frame).shape == (4,84,84):
                tmp = agent.observe(frame)
                tmp.squeeze(0)
                frame = tmp.cpu().numpy()
                del tmp
            # assert(agent.observe(frame).shape == (4,84,84))
            agent.memory_buffer.push(frame, action, reward, next_frame, terminated)
        frame = next_frame

        loss = 0
        if agent.memory_buffer.size() >= learning_start:
            loss = agent.learn_from_experience(batch_size)
            losses.append(loss)

        if i % print_interval == 0:
            print("frames: %5d, reward: %5f, loss: %4f, epsilon: %5f, episode: %4d" % (i, np.mean(all_rewards[-10:]), loss, epsilon, episode_num))
            summary_writer.add_scalar("Temporal Difference Loss", loss, i)
            summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
            summary_writer.add_scalar("Epsilon", epsilon, i)

        if i % update_tar_interval == 0:
            agent.DQN_target.load_state_dict(agent.DQN.state_dict())

        if terminated:

            frame = env.reset()
            first_frame = True
            all_rewards.append(episode_reward)
            episode_reward = 0
            episode_num += 1
            avg_reward = float(np.mean(all_rewards[-100:]))

    summary_writer.close()


    """
    本地展示
    """

    def plot_training(frame_idx, rewards, losses):
        # clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()

    plot_training(i, all_rewards, losses)