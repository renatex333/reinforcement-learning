# rl_compare

O objetivo deste projeto é comparar diversos algoritmos de reinforcement learning considerando diversos ambientes.

Os algoritmos que serão comparados são: 
* DQN
* A2C
* PPO

Vamos utilizar as implementações da biblioteca https://stable-baselines3.readthedocs.io/en/master/

Os ambientes que serão utilizados na comparação são: 
* Bipedal Walker
* Car Racing, discreto e contínuo
* CartPole
* Lunar Lander

Vamos utilizar os ambientes disponibilizados na biblioteca https://gymnasium.farama.org/

## Matriz de comparação

[Documento com a matriz de comparação a ser executada neste projeto](m.pdf)

## Estrutura do repositório

Este repositório está estruturado da seguinte forma: 
* no diretório raiz estão todos os scripts que executam o treinamento, salvam os dados do treinamento e o modelo.
* o diretório **results** deve armazenar todos os arquivos CSV com os dados dos treinamentos.
* o diretório **models** deve armazenar todos os modelos gerados a partir do treinamento. 