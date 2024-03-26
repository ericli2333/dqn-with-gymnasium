import random
import torch
import numpy as np
class replayBuffer(object):
    def __init__(self,
                capacity: int = 100000,
                batch_size:int = 32, 
                ) -> None:
        self.buffer = []
        self.capacity = capacity
        self.curSize = 0
        self.index = 0
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_state(self,observation):
        observation = list(observation)
        state = np.array(observation, dtype=np.float32)
        state = torch.from_numpy(state)
        # state = state.unsqueeze(0)
        state = state.to(self.device)
        return state

        
    def add(self, State, Action, Reward, NextState ,Terminated):
        if type(State) == torch.Tensor:
            State = State.cpu().numpy()
        if type(NextState) == torch.Tensor:
            NextState = NextState.cpu().numpy()
        if type(Action) == torch.Tensor:
            Action = Action.cpu().numpy()
        if type(Reward) == torch.Tensor:
            Reward = Reward.cpu().numpy()
        assert(Reward <= 1 and Reward >= -1)
        record = (State, int(Action), Reward, NextState,Terminated)

        if len(self.buffer) < self.capacity:
            self.buffer.append(record)
        else:
            self.buffer[self.index] = record
            self.index = (self.index + 1) % self.capacity
        self.curSize = len(self.buffer)
        
    def sample(self, batch_size=32):
        '''
        Randomly samples a batch of transitions from the replay buffer.

        Parameters:
            batch_size (int): The number of transitions to sample. Default is 32.

        Returns:
            tuple: A tuple containing the sampled batch of transitions, which includes:
                - states (torch.Tensor): A tensor of shape (batch_size, state_size) containing the states.
                - actions (torch.Tensor): A tensor of shape (batch_size,) containing the actions.
                - rewards (torch.Tensor): A tensor of shape (batch_size,) containing the rewards.
                - next_states (torch.Tensor): A tensor of shape (batch_size, state_size) containing the next states.
                - terminated (torch.Tensor): A tensor of shape (batch_size,) containing the termination flags.
        '''
        states = []
        actions = []
        rewards = []
        next_states = []
        terminated = []
        for i in range(batch_size):
            idx = random.randint(0, self.curSize - 1)
            data = self.buffer[idx]
            state,action,reward,next_state,done = data
            state = self.get_state(state)
            if state.shape == (4,84,84,1):
                state = state.squeeze(3)
            if next_state.shape == (4,84,84,1):
                next_state = next_state.squeeze(3) 
            assert(state.shape == (4,84,84))
            action = int(action)
            next_state = self.get_state(next_state)
            actions.append(action)
            rewards.append(reward)
            states.append(state)
            next_states.append(next_state)
            terminated.append(done)
            # state = self.get_state(self.buffer[idx][0])
            # actions.append(int(self.buffer[idx][1]))
            # rewards.append(self.buffer[idx][2])
            # next_state = self.get_state(self.buffer[idx][3])
            
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int8).long().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.int8).to(self.device)
        next_states = torch.stack(next_states)
        terminated = torch.tensor(terminated, dtype=torch.bool).to(self.device) 
        return (states, actions, rewards, next_states, terminated)

        
if __name__ == '__main__':
    RB = replayBuffer(100000,4)
    for i in range(100):
        RB.add(i, i, i, i)
        
    for _ in range(30):
        batch = RB.sample()
        print(batch)
    