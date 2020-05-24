from dqn_agent import Agent
import numpy as np
import torch
import torch.nn.functional as F


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQN_Agent(Agent):
    def __init__(self, state_size, action_size, seed):       
        super(DDQN_Agent, self).__init__(state_size, action_size, seed)
    
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #simple implementation of a python noob to implement DDQN
        bla = torch.from_numpy(np.zeros(64)).float().to(device)
        for i in range(64):
            bla[i] = self.qnetwork_target(next_states[i]).detach()[self.qnetwork_local(next_states).detach().argmax(1)[i]]
        Q_targets_next = bla.unsqueeze(1)
        #this was my first try of ddqn in python style, but as i said i'm a noob and didn't get it working
        #Q_targets_next = [self.qnetwork_target(next_states).detach()[i] for i in self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                    