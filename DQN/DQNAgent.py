import utility.ReplayBuffer as rb
from utility import QApproximation
from utility.ReplayBuffer import Experience
import torch
import numpy as np

class DQN_agent():
    def __init__(self,
                 in_channels=1, 
                 n_actions = 0, 
                 learning_rate=1e-4, 
                 buffer_size=100000, 
                 epsilon = 0.95,
                 gamma = 0.95
                 ):
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.replay_buffer = rb.ReplayBuffer(capacity=self.buffer_size, batch_size=32)
        self.ValueNetWork = QApproximation.NetWork(in_channels=self.in_channels, action_num=self.n_actions)
        self.optimizer = torch.optim.Adam(self.ValueNetWork.parameters(), lr=self.learning_rate)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.ValueNetWork.to(self.device)
        
    def convert_list_to_tensor(self, list):
        tensor = torch.tensor(list, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        return tensor
    
    def get_action(self, state):
        assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, self.n_actions, (1,))
        else:
            action = self.ValueNetWork(state).argmax()
        return action
    
    def receive_response(self, state:torch.Tensor
                         ,reward
                         ,action
                         ,next_state:torch.Tensor):
        assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        assert(next_state.dtype == torch.float32 and next_state.shape == (1,84,84))
        self.replay_buffer.add(state, action, reward, next_state)
        self.train()

    def train(self):
        if self.replay_buffer.curSize < self.replay_buffer.batch_size:
            return
        samples = self.replay_buffer.sample()
        totalLoss = 0
        for sample in samples:
            # print("1")
            state = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state = sample[3]
            # print(f"before State.shape:{state.shape}")
            # state = self.convert_list_to_tensor(state)
            # print(f"after State.shape:{state.shape}")
            # reward = torch.tensor(reward, dtype=torch.float32)
            next_state = self.convert_list_to_tensor(next_state)
            q_values = self.ValueNetWork(state)[0][action]
            next_q_values = self.ValueNetWork(next_state).max(dim=0).values
            target_q_values = reward + self.gamma * next_q_values
            loss = torch.nn.functional.mse_loss(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            totalLoss += loss.item()
            # print(f'loss: {loss.item()}')
        
        # states = torch.cat(states)
        # actions = torch.cat(actions)
        # rewards = torch.cat(rewards)
        # next_states = torch.cat(next_states)
        # q_values = self.ValueNetWork(states).gather(1, actions)
        # next_q_values = self.ValueNetWork(next_states).max(dim=1).values    
        # target_q_values = rewards + self.gamma * next_q_values
        # loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        return loss
        