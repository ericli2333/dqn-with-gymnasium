import torch
from utility.ReplayBuffer import replayBuffer as rb

RB = rb(1000,4)
tensor = torch.tensor([1,2,3,4], dtype=torch.float32)
for _ in range(100):
    RB.add(1, 2, 3, 4)
    
out = RB.sample()
print(out)
a1 = [row[0] for row in out]
print(a1)