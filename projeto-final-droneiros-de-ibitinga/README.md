# Final Reinforcement Learning Project
# Droneiros de [Ibitinga](https://maps.app.goo.gl/hxKvBDDiBRDcGnjB8)

Welcome to the Final Project of the Reinforcement Learning Course by the Droneiros de Ibitinga group, in which we explored the the drone swarm search problem using Reinforcement Learning algorithms.

## Our Contributors
<div align="center">
<table>
  <tr>
    <td align="center"><a href="https://github.com/RicardoRibeiroRodrigues"><img src="https://avatars.githubusercontent.com/RicardoRibeiroRodrigues" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Ricardo Ribeiro Rodrigues</b></sub></a><br />Developer</td>
    <td align="center"><a href="https://github.com/Pedro2712"><img src="https://avatars.githubusercontent.com/Pedro2712" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Pedro Andrade</b></sub></a><br />Developer</td>
    <td align="center"><a href="https://github.com/JorasOliveira"><img src="https://avatars.githubusercontent.com/JorasOliveira" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Jorás Oliveira</b></sub></a><br />Developer</td>
    <td align="center"><a href="https://github.com/renatex333"><img src="https://avatars.githubusercontent.com/renatex333" width="100px;" alt="" style="border-radius: 50%;"/><br /><sub><b>Renato Laffranchi</b></sub></a><br />Developer</td>
  </tr>
</table>
</div>

## Overview
In this project, we engage with the following environment:

* Drone Swarm Search Environment from [Insper](https://pfeinsper.github.io/drone-swarm-search/)

<p align="center">
  <img src="https://raw.githubusercontent.com/PFE-Embraer/drone-swarm-search/env-cleanup/docs/gifs/render_with_grid_gradient.gif" alt="Drone Swarm Search Example Gif" width="50%">
</p>

This project explores the effectiveness of Reinforcement Learning (RL) algorithms in search-and-rescue operations, specifically using drones to locate a person-in-water (PIW). We aim to investigate various hypotheses regarding the performance of RL compared to greedy algorithms, the convergence rates of agents trained with independent neural networks, the impact of information exchange among agents, and the efficiency of agent coordination in covering search areas. Through these investigations, the project will contribute to the development of more efficient autonomous search strategies in rescue scenarios.

### Hypotheses

**Hypothesis 1:** *Does reinforcement learning (RL) outperform a greedy strategy when the person in water moves away from the region of highest probability?*

To test this hypothesis, we will compare the performance of agents controlled by centralized RL algorithms, such as PPO and DQN, against a greedy policy under the specified base configuration. Performance metrics will include search efficiency, time to locate the PIW, and adaptability to the PIW's movement.

*Expected Outcome:* RL algorithms are anticipated to demonstrate superior adaptability and performance over the greedy approach, particularly as the PIW's location deviates from predicted regions.

**Hypothesis 2:** *Do agents trained with independent neural networks achieve faster convergence compared to those trained with shared networks?*

We will analyze the learning curves of agents in the base configuration using both a centralized learning approach and an independent learning approach. Metrics such as the rate of convergence and final policy performance will be evaluated.

*Expected Outcome:* Preliminary data suggests that the independent approach may converge faster than centralized training. However, literature indicates that centralized learning can converge more quickly and achieve better policy performance. This experiment aims to determine the optimal configuration for our problem and setup.

**Hypothesis 3:** *What is the impact of trajectory information sharing among agents on search operations' effectiveness?*

Using the centralized version of the PPO algorithm, we will test two approaches in an environment with large dispersion:
1. Modifying observations with the agent's trajectory and directly altering probabilities on the matrix.
2. Utilizing a Long Short-Term Memory (LSTM) network to handle sequential information regarding agents' position trajectories.

*Expected Outcome:* Sharing historical search data among agents is predicted to enhance search efficiency through a more coordinated and informed approach. The use of LSTM is expected to boost performance in partially observable environments.

**Hypothesis 4:** *Can agents be organized to ensure full coverage is done in minimal time, while prioritizing the cells of highest probability, thereby maximizing coverage efficiency and minimizing search time?*

This hypothesis will be explored in a coverage environment with 2 drones. The focus will be on evaluating how RL can optimize drone movements to avoid redundant searches and prioritize high-probability regions.

*Expected Outcome:* RL is expected to effectively organize drone movements, reducing search mission time and prioritizing areas with a higher chance of containing PIWs.

**Hypothesis 5:** *How do agents develop their search patterns when there is more than one PIW?*

We will maintain the base configuration while increasing the number of PIWs to four. The search patterns and task distribution among drones will be analyzed.

*Expected Outcome:* The drones are expected to organize themselves to divide the search areas efficiently, increasing the coverage area and enhancing the likelihood of detecting PIWs, thereby improving the overall success rate.

<!--

## Hipóteses

Hipótese 1: o RL consegue ter desempenho superior ao greedy em uma situação onde o náufrago sai da região de maior probabilidade?

-> 4 drones, 1 náufrago, dispersão pequena, POD=1, algoritmo DQN centralizado e independente vs Greedy, Grid 20x20 

Hipótese 2: Agentes treinados com redes neurais independentes convergem mais rápido?

-> 4 drones, 1 náufrago, dispersão pequena x grande, POD=1, algoritmo DQN centralizado e independente, Grid 20x20

Hipótese 3: Qual o impacto da troca de informação entre os agentes? Informação: por onde cada agente passou, poz busca

-> 4 drones, 1 náufrago, dispersão grande, POD=1, algoritmo DQN centralizado e independente, Grid 20x20

-> Dica: DQN com LSTM

Hipótese 4: Agentes conseguem se organizar para buscar em células de uma região uma única vez, todas as células o mais rápido possível

-> 4 drones, náufrago e POD não importam, dispersão grande, algoritmo DQN centralizado e independente, Grid 20x20

## Hypothesis

Hypothesis 1: Does RL outperform greedy in a scenario where the PIW (person-in-water) moves out of the region of highest probability?

Environment Configuration: 4 drones, 1 PIW, small dispersion, POD=1, centralized and independent DQN algorithm vs Greedy, Grid 20x20

Hypothesis 2: Do agents trained with independent neural networks converge faster?

Environment Configuration: 4 drones, 1 PIW, small vs. large dispersion, POD=1, centralized and independent DQN algorithm, Grid 20x20

Hypothesis 3: What is the impact of information exchange among agents? Information: where each agent has been, search area

Environment Configuration: 4 drones, 1 PIW, large dispersion, POD=1, centralized and independent DQN algorithm, Grid 20x20. Tip: DQN with LSTM

Hypothesis 4: Can agents organize to search cells in a region only once, covering all cells as quickly as possible?

Environment Configuration: 4 drones, PIW and POD do not matter, large dispersion, centralized and independent DQN algorithm, Grid 20x20

## Algorithms

In this project, we have implemented ... algorithms.

## Results

### Conclusion

Through rigorous training and evaluation, we observed ..., demonstrating that, ... . However, ... .

Contrastingly, ... . This indicates ... .

Moreover, ... .

In conclusion, our project underscores ... .

-->

## How to reproduce the experiments

To reproduce the tests, follow the instructions on [README experiments](/experiments/README.md)

## References

David Silver, Satinder Singh, Doina Precup, Richard S. Sutton. (2021). [Reward is enough](https://doi.org/10.1016/j.artint.2021.103535).

Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015). [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236).

van Hasselt, Hado and Guez, Arthur and Silver, David. (2016). [Deep Reinforcement Learning with Double Q-Learning](https://ojs.aaai.org/index.php/AAAI/article/view/10295).
