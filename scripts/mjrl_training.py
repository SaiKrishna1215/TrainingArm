#!/usr/bin/env python
# coding: utf-8

# <h1> MJRL training </h1>
# MJRL Algorithms: TRPO - MBAC - PPO - NPG - DAPG <br>
# source: https://github.com/aravindr93/mjrl/tree/master/mjrl/algos

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import gym
from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP, RNN
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.ppo_clip import PPO
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import myosuite
import torch


# <h2> Setting up GPU Usage

# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[3]:


# Create the environment
# Find tasks in: \envs\myo\myobase or \envs\myo\myochallenge
env = gym.make('CenterReachOut-v0')
env.reset()


# <h1> Hyperparameter tuning </h1>
# Still working on it

# In[4]:


import numpy as np
from sklearn.model_selection import ParameterSampler

param_grid = {
    'policy_size': [(256, 256), (128, 128), (64, 64)],
    'rl_step_size': np.logspace(-4, 0, num=10),
    'learnrate': np.logspace(-4, 0, num=10),
    'entcoeff': np.logspace(-4, 0, num=10),
    'batchsize': [16, 32, 64, 128],
    'num_epoch': [5, 10, 20],
    'vf_hidden_size': [(256, 256), (128, 128), (64, 64)],
    'regcoef': np.logspace(-4, 0, num=10),
    'gamma': np.linspace(0.9, 0.999, num=10),
    'gae_lambda': np.linspace(0.9, 1.0, num=10)
}

n_iter_search = 50  

random_search = list(ParameterSampler(param_grid, n_iter=n_iter_search, random_state=123))


# In[ ]:


#Policy hyperparameters
policy_size = (64, 64)  # Size of the policy network
seed = 1  # Random seed used for reproducibility
rl_step_size = 0.1
learnrate = 0.00025
entcoeff = 0.1
batchsize = 16
num_epoch = 10

#Value Function hyperparameters
vf_hidden_size = (256, 256) # Value Function Network size
seed = 1  # Random seed used for reproducibility
rl_step_size = 0.1
learnrate = 0.000325
entcoeff = 0.1
batchsize = 32
num_epoch = 10
regcoef = 1e-3

#Training hyperparameters
iterations = 1000
gamma = 0.995
gae_lambda = 0.97


# Wrap the environment
e = GymEnv(env)


# <H1>Set Policies </h1>

# <h2> For CenterReachOut: </h2>
#     - <b> The observation dimension </b> is: the sensory information (Time, joint Positions, Joint Velocities, Object Position, Palm Position etc... ) <br>
#     - <b> The action dimension </b> is the output of the Policy NN and represenents the 63 values of muscle actuator intensities
# 
# <h2> Display Activations:</h2> 
#     - When the code down below will run, it will automatically generate a tensorboard event that registers the Policy Network. 
#     

# In[10]:


# Initialize policy and baseline
policy = MLP(e.spec, hidden_sizes=policy_size, seed=seed, init_log_std=0.25, min_log_std=1.0)
baseline = MLPBaseline(e.spec, reg_coef=regcoef, batch_size=batchsize , hidden_sizes=vf_hidden_size, epochs=num_epoch, learn_rate=learnrate)


print(e.observation_dim) #For CenterReachOut-v0: 157 
print(e.action_dim) #For CenterReachOut-v0: 63 
exit
# Initialize NPG/PPO agent
agent = NPG(e, policy, baseline, normalized_step_size=rl_step_size, seed=seed, save_logs=True,ent_coeff=entcoeff ,tensorboard_log="./ppo_objhold_tensorboard/")


# In[11]:


print("========================================")
print("Starting policy learning")
print("========================================")

# Train the agent
final_score = train_agent(job_name='.',
            agent=agent,
            seed=seed,
            niter=2500,
            gamma=gamma,
            gae_lambda=gae_lambda,
            sample_mode="trajectories",
            num_traj=96,
            num_samples=0,
            save_freq=50,
            evaluation_rollouts=10)

print("========================================")
print("Job Finished.")
print("========================================")


# <h1> Logging hyperparameters

# In[ ]:


import logging

# Configure the logging system
logging.basicConfig(
    filename=r'C:\Users\Elyas\OneDrive - The University of Colorado Denver\Desktop\Research 2024\Research_papers/training_hyperparameters.log',  # Log file name
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
)


# In[ ]:


# Policy Network Hyperparameters
policy_hyperparameters = {
    'policy_size': (256, 256),
    'seed': 123,
    'rl_step_size': 0.1,
    'learnrate': 0.0025,
    'entcoeff': 0.1,
    'batchsize': 32,
    'num_epoch': 10,
}

# Value Function Network Hyperparameters
vf_hyperparameters = {
    'vf_hidden_size': (256, 256),
    'seed': 123,
    'rl_step_size': 0.1,
    'learnrate': 0.000325,
    'entcoeff': 0.1,
    'batchsize': 32,
    'num_epoch': 10,
    'regcoef': 1e-3,
}

# Training Hyperparameters
training_hyperparameters = {
    'iterations': 1000,
    'gamma': 0.995,
    'gae_lambda': 0.97,
}

# Log Hyperparameters
def log_hyperparameters():
    logging.info("Policy Network Hyperparameters:")
    for key, value in policy_hyperparameters.items():
        logging.info(f"{key}: {value}")
    
    logging.info("\nValue Function Network Hyperparameters:")
    for key, value in vf_hyperparameters.items():
        logging.info(f"{key}: {value}")
    
    logging.info("\nTraining Hyperparameters:")
    for key, value in training_hyperparameters.items():
        logging.info(f"{key}: {value}")

# Call the function to log hyperparameters
log_hyperparameters()

