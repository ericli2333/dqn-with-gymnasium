import utility.ReplayBuffer as rb
from utility import QApproximation
import torch
import numpy as np
import os

class DQN_agent():
    def __init__(self,
                 in_channels=1, 
                 n_actions = 0, 
                 learning_rate=2.5e-4, 
                 buffer_size=100000, 
                 epsilon = 0.95,
                 gamma = 0.95
                 ):
        self.in_channels = in_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.replay_buffer = rb.replayBuffer(capacity=self.buffer_size, batch_size=32)
        self.ValueNetWork = QApproximation.NetWork(in_channels=self.in_channels, action_num=self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.ValueNetWork.parameters(), lr=self.learning_rate)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.ValueNetWork.to(self.device)

    def get_state(self,observation):
        state = np.array(observation)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        state = state.to(self.device)
        return state
    
    def get_action(self, state):
        assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, self.n_actions, (1,))
        else:
            with torch.no_grad():
                # print(state.shape)
                state = state.repeat(32,1,1,1)
                output = self.ValueNetWork(state).detach()
                # print(output)
                action = output.argmax(dim=1)
                # print(action)
                # input('Press Enter to continue...')
        # print(action)
        return action
    
    def receive_response(self, state:torch.Tensor
                         ,reward
                         ,action
                         ,next_state:torch.Tensor):
        assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        assert(next_state.dtype == torch.float32 and next_state.shape == (1,84,84))
        self.replay_buffer.add(state, action, reward, next_state)
        # for _ in range(4):
        self.train()

    def train(self):
        if self.replay_buffer.curSize < self.replay_buffer.batch_size:
            return
        states, rewards, actions, next_states = self.replay_buffer.sample()
        # print(transitions[0])
        # input('Press Enter to continue...')
        # states = [row[0] for row in transitions]
        # rewards = [row[1] for row in transitions]
        # print(rewards)
        # input('Press Enter to continue...')
        # actions = [row[2] for row in transitions]
        # next_states = [row[3] for row in transitions]
        # states = torch.cat(states)
        # states = states.unsqueeze(1)
        # # rewards = torch.stack(rewards)
        # # actions = torch.stack(actions)
        # next_states = torch.cat(next_states)
        # next_states = next_states.unsqueeze(1)
        # print(states.shape)
        Q_values = self.ValueNetWork(states)
        next_Q_values = self.ValueNetWork(next_states).max(dim=1)[0]
        # rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        expected_Q_values = rewards + self.gamma * next_Q_values
        # values = []
        # for estValue, action in zip(Q_values, actions):
            # values.append(estValue[int(action)])
        # values = torch.stack(values)
        values = Q_values[range(states.shape[0]),actions.long()]
        # print(Q_values.shape,values.shape, expected_Q_values.shape)
        # os.pause()
        loss = torch.nn.functional.mse_loss(values, expected_Q_values)
        self.optimizer.zero_grad()
        loss.backward
        self.optimizer.step()
        return loss
        