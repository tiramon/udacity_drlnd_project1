import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN_PER_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):

        # Save experience in replay memory
        d = abs(self.computeQdelta(state, action, reward, next_state, done, GAMMA)[0][0].item())
        self.memory.add(state, action, reward, next_state, done, d)
        #print("mem len",len(self.memory))
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                #print("memory")
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def computeQdelta(self, state, action, reward, next_state, done, gamma):
        # Get max predicted Q values (for next states) from target model
        
        states =  torch.from_numpy(np.vstack([state])).float().to(device)
        actions =  torch.from_numpy(np.vstack([action])).long().to(device)
        rewards =  torch.from_numpy(np.vstack([reward])).float().to(device)
        next_states = torch.from_numpy(np.vstack([next_state])).float().to(device)
        dones = torch.from_numpy(np.vstack([done]).astype(np.uint8)).float().to(device)
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        #  tensor([[0.000]])       
        return Q_targets - Q_expected
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, expIndices = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        deltas = Q_targets-Q_expected
        
        
        for k, expIndex in enumerate(expIndices):
            d = abs(deltas[k][0].item())
            self.memory.memory[expIndex] = self.memory.memory[expIndex]._replace(delta = d, deltapow = d**self.memory.a)
            
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    # by now unused
    e = 0.001
    # used for sampling probability 0 = uniform random, 1 = priority greedy
    a = 0.1
    b = 1.0

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "delta", "deltapow"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done, delta):
        """Add a new experience to memory."""
        delta += self.e
        exp = self.experience(state, action, reward, next_state, done, delta, delta**self.a)
        self.memory.append(exp)
    
    def updatevalue(self, experience): 
        return (1/buffer_size * 1/probabilityValue(experience.deltapow))**b
    
    def probabilityValue(self, deltapow):        
        return deltapow/self.s
        
    def sample(self):
        
        """Randomly sample a batch of experiences from memory."""        
        self.s = sum([entry.deltapow for entry in self.memory])        
        probability = [self.probabilityValue(e.deltapow) for e in self.memory]
        indices = np.random.choice(np.arange(len(self.memory)), self.batch_size, p=probability)
        experiences = [self.memory[i] for i in indices]
      
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        expIndices = [e for e in indices if e is not None]
  
        return (states, actions, rewards, next_states, dones, expIndices)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)