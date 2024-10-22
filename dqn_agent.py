import numpy as np
import random
import torch
import torch.nn as nn
import copy
from collections import deque

from agent import Agent

class DQNAgent(Agent):

    algorithm_name : str = "dqn"

    discount_rate : float = 0.98
    learning_rate : float = 0.01
    epsilon : float = 0.05

    memory_capacity : int = 2048
    batch_size : int = 32

    sync_frequency : int = 128

    is_double_dqn : bool = True

    def __init__(self, input_length, action_count, device):
        self.input_length = input_length
        self.action_count = action_count
        self.device = device

        self.train_count = 0
        self.main_net = self.get_network()
        self.target_net = copy.deepcopy(self.main_net)
        self.optimiser = torch.optim.Adam(self.main_net.parameters(), lr = self.learning_rate)
        self.memory = deque(maxlen=self.memory_capacity)

        self.observation = None
        self.pre_observation = None
    
    def get_network(self):
        network = nn.Sequential(
            nn.Linear(self.input_length, 64), nn.ReLU(),
            #nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(64, self.action_count)
        )
        return network

    def get_action(self, observation):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_count - 1)
        else:
            status = self.get_status(observation)
            with torch.no_grad():
                if self.is_double_dqn:
                    q = self.main_net(status).detach().numpy()                
                else:
                    q = self.target_net(status).detach().numpy()
            action = np.argmax(q)
        self.action = action
        self.observation = observation
        return action

    def get_status(self, observation):
        if observation is None: return None
        return torch.tensor(observation)

    def train(self, observation, action, reward, next_observation):
        super().train(observation, action, reward, next_observation)
        self.memory_append(observation, action, reward, next_observation)
        
        if len(self.memory) >= self.batch_size:
            self.replay()

        if (self.train_count + 1) % self.sync_frequency == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
        self.train_count += 1

        return None
    
    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        losses = None
        for status, action, reward, next_status in batch:
            with torch.no_grad():
                if next_status is None:
                    max_q = torch.tensor(0).detach()
                else:
                    max_q = torch.tensor(self.main_net(next_status).detach().numpy().max()).detach()
            target_q = reward + self.discount_rate * max_q
            main_q = self.main_net(status)[action]
            loss = nn.functional.mse_loss(main_q.float(), target_q.float())
            if losses is None:
                losses = loss
            else:
                losses += loss
        losses /= self.batch_size
        self.optimiser.zero_grad()
        losses.backward()
        self.optimiser.step()

    def memory_append(self, observation, action, reward, next_observation):
        self.memory.append((
            self.get_status(observation),
            action,
            reward,
            self.get_status(next_observation)
        ))

    def train_brief(self, reward):
        if self.observation is not None and self.pre_observation is not None:
            self.train(self.pre_observation, self.action, reward, self.observation)

        self.observation = None
        self.pre_observation = self.observation


    def load_model(self, name:str):
        self.main_net = torch.load(f'./models/{name}_main.pth')
        self.target_net = torch.load(f'./models/{name}_target.pth')

    def save_model(self, path:str, name:str):
        torch.save(self.main_net, f'{path}{name}_main.pth')
        torch.save(self.target_net, f'{path}{name}_target.pth')