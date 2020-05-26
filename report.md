Current results of my DQNs can be found at https://docs.google.com/spreadsheets/d/1eUB7RycHVyyxyvEqwnS6_q2bdTcYS8Fv6HvPdg88EAw/edit?usp=sharing

# Training an agent with deep Q-learning

* [Neural Network Layout](#layout)
* [Training](#training)
* [Future improvements](#future_improvements)

<a name="layout"></a>
## Neural Network Layout

The layout of the neural network used by all agents is described below and the code can be found [here](./model.py)

| Layer  | In | Out | Activation |
|--------|----|----:|------------|
| Linear | 37 | 64  | RELU       |
| Linear | 64 | 64  | RELU       |
| Linear | 64 |  4  | &nbsp;     |

## Agents
I tried multiple aproaches. First i tried a normal deep Q-network with fixed q targets and memory replay from previous lessons adapted to the current environment. [Code](./dqn_agent.py)

After that i tried another aproache and wanted to try if a double deep Q-network with memory replay would deliver faster/better results. [Code](./ddqn_agent.py)

<a name="training"></a>
## Training

The agent has been trained with a double deep q-network with memory replay until it reached a average score of +13 over 100 consecutive episodes as required by the udacity nanodegree.

### Parameters
#### Memory Replay
| Parameter | Value | Description |
|-----------|-------:|---|
| Buffer size | 10000 |
| Batch size | 64 |
| update every x turns | 4 |
| learning rate | 0,0005 |

| Parameter | Value | Description |
|-----------|-------:| ---|
| tau | 0.003 | Used by Fixed Q and Double DQN
| gamma | 0.99 |
| start epsilon | 1.0 |
| minimal epsilon | 0.0 |
| epsilon decay | 0.995 |

Optimizer = Adam

### Training progress
![](./ddqn_trained_496_episodes.png =20x)

### Trained weights
Stored weights can be found at [ddqn_trained_496.pth](ddqn_trained_496.pth)

<a name="future_improvements"></a>
## Future improvements


