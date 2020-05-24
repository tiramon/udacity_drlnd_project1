a README that describes how someone not familiar with this project should use your repository. The README should be designed for a general audience that may not be familiar with the Nanodegree program; you should describe the environment that you solved, along with how to install the requirements before running the code in your repository.

# Project: Navigation
This project is about training an agent to to navigate in a large square world and gather yellow bannas.

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
The project environment is similar but not identical to the Banana Collector of Unity ML-Agents
The Environment can be downloaded at
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

### Dependencies
Dependencies needed to get the programm running are gathered in the requirements.txt to install those execute the command:

`` pip install requirements.txt

To train the agent either run
`` python run.py 

or execute TainBot.ipynb in jupyter

## Files
Agents:
* dqn_agent.py - a DQN agent with Memory replay und Fixed Q-Targets
* ddqn_agent.py - a double DQN agent with Memory replay

* model.py - the configuration of the neural network

* run.py - training the agent in the command line
* TrainBot.ipynb - training the agent in a jupyter notebook

* TrainedBot.ipynb - running the agent with already created weights to validate previous training