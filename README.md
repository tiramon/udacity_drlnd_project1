# Project: Navigation
This project is about training a deep reinforcment learning agent to navigate in a large square world and gather yellow bannas, without the agent knowing anything about the environment or rules.

## About the Environment

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Setting up the environment

### Download environment
The project environment is similar but not identical to the Banana Collector of Unity ML-Agents
The Environment can be downloaded at

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

### Install dependencies
Dependencies needed to get the programm running are gathered in the requirements.txt to install those execute the command:

```bash
 pip install requirements.txt
```

I had problems to install torch==0.4.0 so if you execute 

```bash
pip install torch==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html 
```

before the command above it should work as expected

## Run Agent

To train the agent either run

```bash
python run.py 
```

or execute TrainBot.ipynb in jupyter

To see how a trained bot scores execute TrainedBot.ipynb in jupyter

## Files
Agents:

* dqn_agent.py - a DQN agent with Experience Replay and Fixed Q-Targets |
* ddqn_agent.py - a double DQN agent with Experience Replay |
* dqn_per_agent.py - a DQN agent with Prioritized Experience Replay and Fixed Q-Targets |

Network Structure:

* model.py - the configuration of the neural network

Files to train:

* run.py - training the agent in the command line
* TrainBot.ipynb - training the agent in a jupyter notebook

Files to validate:

* TrainedBot.ipynb - running the agent with already created weights to validate previous training