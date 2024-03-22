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
                 epsilon = 0.9,
                 gamma = 0.99,
                 log_level = 1,
                 ):
        self.in_channels = in_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.log_level = log_level
        self.replay_buffer = rb.replayBuffer(capacity=self.buffer_size, batch_size=32)
        self.ValueNetWork = QApproximation.NetWork(in_channels=self.in_channels, action_num=self.n_actions)
        self.optimizer = torch.optim.Adam(self.ValueNetWork.parameters(), lr=self.learning_rate)
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
        # print(f"state: {state}")
        # assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        if torch.rand(1) > self.epsilon:
            action = torch.randint(0, self.n_actions, (1,))
        else:
            if type(state) != torch.Tensor:
                state = self.get_state(state)
            with torch.no_grad():
                state = state.repeat(32,1,1,1)
                output = self.ValueNetWork(state).detach()
                action = output.argmax(dim=1)
                del output
                # torch.cuda.empty_cache()
        return action
    
    def receive_response(self, state
                         ,reward
                         ,action
                         ,next_state):
        # assert( state.shape == (1,84,84))
        # assert( next_state.shape == (1,84,84))
        self.replay_buffer.add(state, action, reward, next_state)
        self.train()

    def train(self):
        if self.replay_buffer.curSize < self.replay_buffer.batch_size:
            return (0)
        states, rewards, actions, next_states = self.replay_buffer.sample()
        Q_values = self.ValueNetWork(states)
        next_Q_values = self.ValueNetWork(next_states).max(dim=1)[0]
        expected_Q_values = rewards + self.gamma * next_Q_values
        values = Q_values[range(states.shape[0]),actions.long()]
        if self.log_level == 2:
            print(f"values: {values}\nexpected_Q_values: {expected_Q_values}")
            for name, parms in self.ValueNetWork.named_parameters():	
                    print('-->name:', name)
                    print('-->para:', parms)
                    print('-->grad_requirs:',parms.requires_grad)
                    print('-->grad_value:',parms.grad)
                    print("===")
            input("Press Enter to continue...")
        loss = torch.nn.functional.smooth_l1_loss(values, expected_Q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # del Q_values, next_Q_values, expected_Q_values, values
        return loss
        