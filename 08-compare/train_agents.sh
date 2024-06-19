#!/bin/bash

# Carpole environment
cd src
nohup python cart_pole_dqn.py &
nohup python cart_pole_ppo_env_1.py &
nohup python cart_pole_ppo_env4.py &
nohup python cart_pole_ppo_env8.py &
nohup python cartpole_a2c_env1.py & 

# bipedal walker environment
nohup python bipedal_walker_a2c.py &
nohup python bipedal_walker_ppo_hugging_face.py &
nohup python bipedal_walker_ppo.py & 

#lunarlander environment
nohup python lunar_lander_a2c_env1.py & 
nohup python lunar_lander_dqn.py &
nohup python lunar_lander_ppo_env_1.py &
nohup python lunarlander_a2c_env_8.py & 

#car racing discreto environment
nohup python car_racing_discrete_cnn_A2C_env_8.py & #tem interacao com usuario
nohup python car_racing_discreto_cnn_ppo_env_8.py & #tem interacao com usuario
nohup python car_racing_discreto_mlp_ppo_env8.py & #tem interacao com usuario
nohup python car_racing_discrete_cnn_ppo_env1.py &   
nohup python car_racing_discreto_mlp_a2c.py &        
nohup python car_racing_discreto_mlp.py &
nohup python car_racing_discreto_cnn_dqn.py &        
nohup python car_racing_discreto_mlp_ppo_env_1.py & 

#car racing continuo environment
nohup python car_racing_continous_ppo_mlp_1_env.py &
nohup python car_racing_continuos_cnn_ppo_env_1.py & #tem interacao com usuario





