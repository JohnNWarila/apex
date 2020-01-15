from rl.policies.actor import Gaussian_FF_Actor
from rl.policies.critic import FF_V
from rl.policies.gaussian_mlp import GaussianMLP
from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot

import functools

import torch
import copy

import numpy as np
import os
import time

filename = "fwrd_walk_StateEst_speed-05-3_freq1-2"
old_policy = torch.load("./trained_models/{}.pt".format(filename))
old_dict = old_policy.state_dict()

obs_dim = old_policy.state_dict()['actor_layers.0.weight'].shape[1]
act_dim = len(old_policy.state_dict()['mean.bias'])

actor = Gaussian_FF_Actor(obs_dim, act_dim, fixed_std=np.exp(-2), nonlinearity=old_policy.nonlinearity, 
        obs_std=old_policy.obs_std, obs_mean=old_policy.obs_mean, normc_init=False)
critic = FF_V(obs_dim, nonlinearity=old_policy.nonlinearity, obs_std=old_policy.obs_std, obs_mean=old_policy.obs_mean, normc_init=False)

act_dict = actor.state_dict()
critic_dict = critic.state_dict()

for key in old_dict.keys():
    if "actor" in key:
        act_dict[key].copy_(old_dict[key].data)
    elif "critic" in key:
        critic_dict[key].copy_(old_dict[key].data)
    elif "mean" in key:
        index = 4
        new_key = key[:index] + 's' + key[index:]
        act_dict[new_key].copy_(old_dict[key].data)
    elif "vf" in key:
        index = 2
        new_key = "network_out" + key[index:]
        # critic_dict[new_key] = copy.deepcopy(old_dict[key])
        critic_dict[new_key].copy_(old_dict[key].data)
        # print("new_key: ", new_key)
    else:
        pass
print("done converting, doing check")
# for key in act_dict.keys():
#     if "actor" in key:
#         print("check {}: {}".format(key, torch.all(act_dict[key] == old_dict[key])))
#     elif "mean" in key:
#         index = 4
#         old_key = key[:index] + key[index+1:]
#         print("old key: ", old_key)
#         print("check {}: {}".format(key, torch.all(act_dict[key] == old_dict[old_key])))
#     else:
#         print("missing key: ", key)
# for key in critic_dict.keys():
#     if "critic" in key:
#         print("check {}: {}".format(key, torch.all(critic_dict[key] == old_dict[key])))
#     elif "network_out" in key:
#         index = 10
#         old_key = "vf" + key[index+1:]
#         print("old key: ", old_key)
#         print("check {}: {}".format(key, torch.all(critic_dict[key] == old_dict[old_key])))
#     else:
#         print("missing key: ", key)

torch.save(actor, "./trained_models/new_policies/"+filename+"_actor.pt")
torch.save(critic, "./trained_models/new_policies/"+filename+"_critic.pt")