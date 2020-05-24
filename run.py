from unityagents import UnityEnvironment
import numpy as np
from collections import deque

env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=True)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

action_size = brain.vector_action_space_size
env_info = env.reset(train_mode=True)[brain_name]
state = env_info.vector_observations[0]
state_size = len(state)

from ddqn_agent import DDQN_Agent
agent = DDQN_Agent(state_size, action_size, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    moves = []                        # list containing scores from each episode
    moves_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                   # initialize epsilon
    actionC = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]       
        score = 0
        move = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0] 
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            move += 1
            if done:
                break 
                
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        moves_window.append(move)       # save most recent score
        moves.append(move)         
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} Average Moves: {:.2f}'.format(i_episode, np.mean(scores_window),np.mean(moves_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} Average Moves: {:.2f}'.format(i_episode, np.mean(scores_window),np.mean(moves_window)))
        if np.mean(scores_window)>=13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f} Average Moves: {:.2f}'.format(i_episode-100, np.mean(scores_window),np.mean(moves_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_trained_{:d}_episodes.pth'.format(i_episode))
            break
    
    return scores

scores = dqn(1000)
env.close()

# plot the scores
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(np.arange(len(scores)), scores)
#plt.ylabel('Score')
#plt.xlabel('Episode #')
#plt.savefig('graph_trained_{:d}_episodes.png'.format(len(scores))