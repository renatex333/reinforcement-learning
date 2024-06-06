import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
import sys

env_name = sys.argv[1]
model_path = sys.argv[2]
algol = sys.argv[3]

env = gym.make(env_name, render_mode='human')

if algol == 'DQN':
    model = DQN.load(model_path)
elif algol == 'PPO':
    model = PPO.load(model_path)
elif algol == 'A2C':
    model = A2C.load(model_path)
else:
    print('Algoritmo não reconhecido')
    exit()

(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

