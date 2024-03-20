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
        self.index = 0
        self.batch_size = batch_size
        
    def add(self, experience:Experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.popleft()
        self.buffer.append(experience)
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch