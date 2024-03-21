from collections import deque, namedtuple
import random
import torch
import numpy as np
class replayBuffer(object):
    def __init__(self,
                capacity: int = 100000,
                batch_size:int = 32, 
                ) -> None:
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.curSize = 0
        self.index = 0
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, State, Action, Reward, NextState):
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append((State, int(Action), Reward, NextState))
        self.curSize = len(self.buffer)
        
    def sample(self,batch_size = 32):
        states = []
        actions = []
        rewards = []
        next_states = []
        for i in range(batch_size):
            idx = random.randint(0, self.curSize - 1)
            states.append(self.buffer[idx][0])
            actions.append(int(self.buffer[idx][1]))
            rewards.append(self.buffer[idx][2])
            next_states.append(self.buffer[idx][3])
            
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int8).long().to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states)
        return (states,actions,rewards,next_states)

        
if __name__ == '__main__':
    RB = replayBuffer(100000,4)
    for i in range(100):
        RB.add(i, i, i, i)
        
    for _ in range(30):
        batch = RB.sample()
        print(batch)
    