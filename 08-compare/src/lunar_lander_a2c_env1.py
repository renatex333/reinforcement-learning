from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
import csv

tmp_path = "../results/lunar_lander_a2c_env1/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("LunarLander-v2")
model = A2C(policy = "MlpPolicy", env = env)

model.set_logger(new_logger)
model.learn(total_timesteps=100000)
model.save("../models/lunar_lander_a2c_env1")


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

print('modelo treinado')
env = gym.make("LunarLander-v2", render_mode='human')

(obs,_) = env.reset()

del model
model = A2C.load("../models/lunar_lander_a2c_env1")


rewards = []
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    rewards.append(reward)

    ####
    if done:
    #   obs = env.reset()
      (obs,_) = env.reset()


# with open(csv_file, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Iteration', 'Reward'])
#     for reward in rewards:
#         writer.writerow(reward)