# src/rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 26)
        )
    def forward(self, x):
        return self.net(x)

class RLAgent:
    def __init__(self, state_dim: int):
        self.model = DQN(state_dim).to(device)
        self.target = DQN(state_dim).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=100_000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        self.update_every = 100

    def act(self, state: np.ndarray, guessed_vec: np.ndarray):
        if random.random() < self.epsilon:
            avail = [i for i, g in enumerate(guessed_vec) if g == 0]
            return random.choice(avail) if avail else 0
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state_t).cpu().detach().numpy()[0]
        q_values[guessed_vec == 1] = -np.inf
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)

        current_q = self.model(states).gather(1, actions).squeeze(1)
        next_q = self.target(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.target.load_state_dict(self.model.state_dict())