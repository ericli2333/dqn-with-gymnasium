from collections import deque
import random
class Experience(object):
    def __init__(self,state,action,reward,nextState) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.terminal = False

class ReplayBuffer(object):
    def __init__(self,
                capacity: int = 100000,
                batch_size:int = 32, 
                ) -> None:
        self.buffer = []
        self.capacity = capacity
        self.curSize = 0
        self.index = 0
        self.batch_size = batch_size
        
    def add(self, State, Action, Reward, NextState):
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append((State, Action, Reward, NextState))
        self.curSize = len(self.buffer)
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch