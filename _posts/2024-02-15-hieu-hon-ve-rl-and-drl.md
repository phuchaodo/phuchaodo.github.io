---
title: '[Note] Hiểu hơn về RL and DRL'
date: 2024-02-15
permalink: /posts/2024/02/15/hieu-hon-ve-rl-and-drl/
tags:
  - research
  - proposal
  - RL
  - DRL
--- 

Hiểu hơn về một số khái niệm và code về RL và DRL

Outline
======

1. Genetic algorithm

2. PSO

3. DRL


Concept RL
======

*Reinforcement* learning (RL) algorithms operate in a unique fashion compared to other learning methods. Here's a breakdown of their input, output, processing, and evaluation:

**Inputs:**

* **Environment:** This can be a real-world system or a simulated one. The agent interacts with the environment, receiving observations about its state (e.g., sensor readings, game board status).
* **Actions:** These are the possible choices the agent can make in the environment.
* **Rewards:** These are numerical signals provided by the environment after the agent takes an action. Positive rewards signify good actions, negative ones punish bad ones.

**Outputs:**

* **Action selection:** The agent's main output is selecting an action based on its current understanding of the environment and its reward goals.
* **Learning updates:** Internally, the agent updates its policy (how it chooses actions) based on the rewards it receives. This allows it to adapt and improve over time.

**Processing:**

* **Policy selection:** The agent's policy determines how it chooses actions. Popular approaches include value-based methods (estimating the value of different states and actions) and policy-based methods (directly learning an action probability distribution).
* **Learning updates:** Based on received rewards and its policy, the agent updates its internal parameters using various algorithms like Q-learning, SARSA, or Deep Q-Networks. These updates aim to improve the policy for future actions.

**Evaluation metrics:**

* **Return:** The total sum of rewards received over an episode (a complete interaction with the environment).
* **Average return:** The average return obtained over multiple episodes, indicating overall performance.
* **Exploration vs. exploitation trade-off:** Balancing exploring new actions to learn the environment versus exploiting known, high-reward actions is crucial. Metrics like exploration rate can assess this balance.
* **Success rate:** In specific tasks, measuring the percentage of successful completions can gauge performance.

**Key points:**

* Reinforcement learning is trial-and-error based, relying on rewards to guide learning.
* The agent interacts with the environment, learns from feedback, and improves its policy.
* Evaluation metrics assess the learning progress and policy effectiveness.

Remember, the specific details of input, output, processing, and evaluation vary depending on the chosen RL algorithm and the problem setting. This provides a general understanding of how reinforcement learning algorithms work and are evaluated.



RL in RA
======

from this:  [Link](https://github.com/Engineer1999/Double-Deep-Q-Learning-for-Resource-Allocation)

from this:  [Link](https://github.com/gundoganalperen/DIRAL/tree/main)

from this:  [Link](https://github.com/xiangni/DREAM)

from this: 
 [Link](https://github.com/davidtw0320/Resources-Allocation-in-The-Edge-Computing-Environment-Using-Reinforcement-Learning)

from this:  [Link](https://github.com/wn-upf/decentralized_qlearning_resource_allocation_in_wns/tree/master)

from this: code xịn:  [Link](https://github.com/datawhalechina/easy-rl/tree/master/notebooks)

from this:  [Link](https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo)


Link ref
======

[Genetic algorithm](https://colab.research.google.com/drive/1LevKUHVljPtbE6MlBQsGmB2myI8kCdoa?usp=sharing)

[PSO](https://colab.research.google.com/drive/1zqqIyr_R1p07L_2QiLufzQapx2wxqZbS?usp=drive_link)

[DRL](https://colab.research.google.com/drive/1ZQY0O_irqi1FuX57H3IIWwtBgVh7XOdK?usp=drive_link)

[Colab link](https://colab.research.google.com/drive/1_aDTl0VjSezW-syzHssqAK0-JXkboN8_?usp=sharing#scrollTo=3o1-KRDPcncA)

Hết.
