import utility.ReplayBuffer as rb
from utility import QApproximation
from utility.ReplayBuffer import Experience
import torch
import numpy as np
from collections import namedtuple
Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))

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

    def get_state(self,observation):
        state = np.array(observation)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        return state
    
    def get_action(self, state):
        assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, self.n_actions, (1,))
        else:
            action = self.ValueNetWork(state).detach().max(dim=1)[1].view(1,1)
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
        transitions = self.replay_buffer.sample()
        totalLoss = 0
        batch =  Transition(*zip(*transitions))
        actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))
        return totalLoss
        