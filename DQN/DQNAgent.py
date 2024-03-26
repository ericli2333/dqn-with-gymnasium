import utility.ReplayBuffer as rb
from utility import QApproximation
import torch
import numpy as np
import random
import os

class DQN_agent():
    def __init__(self,
                 writer,
                 in_channels=4, 
                 n_actions = 0, 
                 learning_rate=3e-4, 
                 buffer_capacity=100000, 
                 epsilon = 0.9,
                 gamma = 0.99,
                 log_level = 1,
                 ):
        self.writer = writer
        self.in_channels = in_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 定义设备
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.buffer_capacity = buffer_capacity
        self.epsilon = epsilon
        self.gamma = gamma
        self.log_level = log_level
        self.replay_buffer = rb.replayBuffer(capacity=self.buffer_capacity, batch_size=32)
        self.ValueNetWork = QApproximation.NetWork(in_channels=self.in_channels, action_num=self.n_actions)
        for param in self.ValueNetWork.parameters():
            param.data.uniform_(1e-7, 3e-7)
        self.optimizer = torch.optim.Adam(self.ValueNetWork.parameters(), lr=self.learning_rate)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.ValueNetWork.to(self.device)

    def get_state(self,observation):
        state = np.array(observation.__array__()[None]/255, dtype=np.float32)
        state = torch.from_numpy(state)
        # state = state.unsqueeze(0)
        state = state.to(self.device)
        return state
    
    def value(self,state):
        return self.ValueNetWork(state)

    def get_action(self, state,epsilon):
        # print(f"state: {state.shape} type: {type(state)}")
        # assert(state.dtype == torch.float32 and state.shape == (1,84,84))
        if torch.rand(1) > epsilon :
            action = int(random.randrange(self.n_actions))
        else:
            # if type(state) != torch.Tensor:
                # state = self.get_state(state)
            with torch.no_grad():
                # state = state.repeat(32,1,1,1)
                state = state.unsqueeze(0)
                # print(f"state: \n{state.shape}, {state.dtype}")
                # input()
                output = self.ValueNetWork(state).cpu().detach().numpy()
                action = int(output.argmax(1)[0])
                del output
                # torch.cuda.empty_cache()
        return action
    
    def receive_response(self, state
                         ,reward
                         ,action
                         ,next_state,
                         terminated : bool):
        # assert( state.shape == (1,84,84))
        # assert( next_state.shape == (1,84,84))
        self.replay_buffer.add(state, action, reward, next_state,terminated)
        # self.train(frame)

    def print_model(self):
        for name, parms in self.ValueNetWork.named_parameters():	
                print('-->name:', name)
                print('-->para:', parms)
                print('-->grad_requirs:',parms.requires_grad)
                print('-->grad_value:',parms.grad)
                print("===")
        input("Press Enter to continue...")

    def calculate_loss(self, states, actions, rewards, next_states, terminated, frame):
        """
        Calculates the loss for the DQN agent.

        Args:
            states (torch.Tensor): The current states.
            actions (torch.Tensor): The actions taken.
            rewards (torch.Tensor): The rewards received.
            next_states (torch.Tensor): The next states.
            terminated (torch.Tensor): A boolean tensor indicating whether the episode terminated.
            frame (int): The current frame.

        Returns:
            torch.Tensor: The calculated loss.

        """
        Q_values = self.ValueNetWork(states)
        values = Q_values[range(states.shape[0]), actions.long()]
        next_Q_values = self.ValueNetWork(next_states).max(-1)[0]
        expected_Q_values = rewards + self.gamma * next_Q_values
        self.writer.add_scalar('values', values.float().mean(), frame)
        # self.writer.add_scalar('rewards', rewards.float().mean(), frame)
        if self.log_level == 2:
            print(f'values: {values}\nexpected_Q_values: {expected_Q_values}')
            input("Press Enter to continue...")
        expected_Q_values = torch.where(terminated, rewards, expected_Q_values)
        loss = torch.nn.functional.mse_loss(values, expected_Q_values.detach())
        return loss
        
    
    def train(self,frame):
        if self.replay_buffer.curSize < self.replay_buffer.batch_size:
            return (0)
        states, actions ,rewards, next_states ,terminated= self.replay_buffer.sample()
        loss = self.calculate_loss(states, 
                                   actions=actions, 
                                   rewards=rewards, 
                                   next_states=next_states,
                                   terminated=terminated,
                                   frame=frame)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.ValueNetWork.parameters():
        #     param.grad.data.clamp_(-1,1)
        torch.nn.utils.clip_grad_norm_(self.ValueNetWork.parameters(), max_norm=20, norm_type=2)

        self.optimizer.step()
        if self.log_level == 2:
            self.print_model()
        if self.log_level == 1:
            self.writer.add_scalar('learning rate',self.optimizer.param_groups[0]['lr'],frame)
            if frame % 1000 == 0:
                for name, param in self.ValueNetWork.named_parameters():
                    self.writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=frame // 1000)
                    self.writer.add_histogram(tag=name+'_data', values=param.data, global_step=frame // 1000)
        # del Q_values, next_Q_values, expected_Q_values, values
        return loss
        