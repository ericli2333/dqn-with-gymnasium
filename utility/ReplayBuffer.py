from collections import deque, namedtuple
import random
Transition = namedtuple('Transion', 
                        ('state', 'action', 'next_state', 'reward'))
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
        
    def add(self, *args):
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append(Transition(*args))
        self.curSize = len(self.buffer)
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        return batch

        
if __name__ == '__main__':
    RB = ReplayBuffer(100000,4)
    for i in range(100):
        RB.add(i, i, i, i)
        
    for _ in range(30):
        batch = RB.sample()
        print(batch)
    