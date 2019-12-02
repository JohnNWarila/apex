import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from rl.utils.remote_replay import ReplayBuffer
from rl.policies.actor import SAC_Gaussian_Actor
from rl.policies.critic import Dual_Q_Critic as Critic

import functools

import ray

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

# Runs policy for X episodes and returns average reward. Optionally render policy
def evaluate_policy(env, policy, eval_episodes=10, max_traj_len=400):
    avg_reward = 0.0
    avg_eplen = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        t = 0
        done_bool = 0.0
        while not done_bool:
            t += 1
            action = policy.select_action(np.array(obs), param_noise=None)
            obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if t + 1 == max_traj_len else float(done)
            avg_reward += reward
        avg_eplen += t

    avg_reward /= eval_episodes
    avg_eplen /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_eplen


"""
This implementation of SAC will only support a stochastic policy for now. SAC with a deterministic actor is very close to TD3, which is already implemented in apex
"""
class SAC():
    def __init__(self, state_dim, action_dim, max_action, a_lr, c_lr, alpha, env_name='NOT_SET'):
        self.actor = SAC_Gaussian_Actor(state_dim, action_dim, env_name=args.env_name, nonlinearity=torch.nn.functional.relu, action_space=None)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, action_dim, hidden_size=256, env_name=env_name).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size=256, env_name=env_name).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.max_action = max_action

        self.alpha = alpha

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        self.actor.eval()

        return self.actor.act(state, deterministic=False).cpu().data.numpy().flatten()

    # TODO: benchmark this and find ways to make more efficient (especially with prepping data for logs)
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005):

        avg_q1, avg_q2, q_loss, pi_loss, ent_loss = (0,0,0,0,0)

        for it in range(iterations):

            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy. unlike td3 don't use actor target and don't add clipped noise
            next_action, action_log_prob = self.actor.sample(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            min_Q = torch.min(target_Q1, target_Q2) - self.alpha * action_log_prob
            target_Q = reward + (done * discount * min_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Keep track of Q estimates for logging
            avg_q1 += torch.mean(current_Q1)
            avg_q2 += torch.mean(current_Q2)
            avg_action += next_action

            # Compute critic loss
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Keep track of Q loss for logging
            q_loss += critic_loss

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = ((self.alpha * action_log_prob) - min_Q).mean()

            # Keep track of pi loss for logging
            pi_loss += actor_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Opinion: Rather than delaying updates by some interval, decreasing tau should be sufficient for tuning the lag between target and main networks

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data)

        # prep info for logging
        avg_q1 /= iterations
        avg_q2 /= iterations
        q_loss /= iterations
        pi_loss /= iterations
        ent_loss = 0 # CHANGE THIS WHEN AUTO ENTROPY ADJUSYMENT ADDED

        return avg_q1, avg_q2, q_loss, pi_loss, ent_loss

    def save(self):
        if not os.path.exists('trained_models/sac/'):
            os.makedirs('trained_models/sac/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.actor, os.path.join(
            "./trained_models/sac", "actor_model" + filetype))
        torch.save(self.critic, os.path.join(
            "./trained_models/sac", "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "actor_model.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor = torch.load(actor_path)
            self.actor.eval()
        if critic_path is not None:
            self.critic = torch.load(critic_path)
            self.critic.eval()

def run_experiment(args):
    from apex import env_factory, create_logger

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror)
    max_traj_len = args.max_traj_len

    # Start ray
    ray.init(num_gpus=0, include_webui=True, redis_address=args.redis_address)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    max_action = 1.0
    #max_action = float(env.action_space.high[0])

    print()
    print("Soft Actor Critic:")
    print("\tenv:            {}".format(args.env_name))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print("\tseed:           {}".format(args.seed))
    # print("\tmirror:         {}".format(args.mirror))
    # print("\tnum procs:      {}".format(args.num_procs))
    print("\tmin steps:      {}".format(args.num_steps))
    print("\ta_lr:           {}".format(args.a_lr))
    print("\tc_lr:           {}".format(args.c_lr))
    print("\alpha:           {}".format(args.alpha))
    print("\ttau:            {}".format(args.tau))
    print("\tgamma:          {}".format(args.discount))
    print("\tbatch size:     {}".format(args.batch_size))
    print()

    # Initialize policy, replay buffer
    policy = SAC(state_dim, action_dim, max_action, a_lr=args.a_lr, c_lr=args.c_lr, alpha=args.alpha, env_name=args.env_name)

    replay_buffer = ReplayBuffer()

    # create a tensorboard logging object
    logger = create_logger(args)

    total_timesteps = 0
    total_updates = 0
    timesteps_since_eval = 0
    episode_num = 0
    
    # Evaluate untrained policy once
    ret, eplen = evaluate_policy(env_fn(), policy)
    logger.add_scalar("Test/Return", ret, total_updates)
    logger.add_scalar("Test/Eplen", eplen, total_updates)

    policy.save()

    while total_timesteps < args.max_timesteps:

        # TODO: Parallelize experience collection
        # collect parallel experience and add to replay buffer
        env = env_fn()

        # reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        while not done and episode_timesteps < max_traj_len:

            # select action randomly if at start of training
            if total_timesteps < args.start_timesteps:
                # action = torch.randn(self.env.action_space.shape[0])  # do this for CassieEnv
                action = env.action_space.sample()
            else:
                # select action from policy
                action = policy.select_action(obs)

            # Perform action
            new_obs, reward, done, _ = env.step(action)
            done_bool = 1.0 if episode_timesteps + 1 == max_traj_len else float(done)
            episode_reward += reward

            # Store data in replay buffer
            transition = (obs, new_obs, action, reward, done_bool)
            replay_buffer.add(transition)

            # update state
            obs = new_obs

            # increment counters
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        # episode is over now

        # Logging rollouts
        print("Total T: {}\tEpisode Num: {}\tEpisode T: {}\tReward: {}".format(total_timesteps, episode_num, episode_timesteps, episode_reward))

        # update the policy
        avg_q1, avg_q2, q_loss, pi_loss, ent_loss = policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
        total_updates += episode_timesteps      # this is how many iterations we did updates for

        # Logging training
        logger.add_scalar("Train/avg_q1", avg_q1, total_updates)
        logger.add_scalar("Train/avg_q2", avg_q2, total_updates)
        logger.add_scalar("Train/q1_loss", q_loss, total_updates)
        logger.add_scalar("Train/pi_loss", pi_loss, total_updates)
        logger.add_scalar("Train/ent_loss", ent_loss, total_updates)
        # logger.add_scalar("Train/alpha", alpha, total_updates)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval = 0
            ret, eplen = evaluate_policy(env_fn(), policy)

            # Logging Eval
            logger.add_scalar("Test/Return", ret, total_updates)
            logger.add_scalar("Test/Eplen", eplen, total_updates)

            # Logging Totals
            logger.add_scalar("Misc/Timesteps", total_timesteps, total_updates)
            logger.add_scalar("Misc/ReplaySize", replay_buffer.ptr, total_updates)

            print("Total T: {}\tEval Eplen: {}\tEval Return: {} ".format(total_timesteps, eplen, ret))

            if args.save_models:
                policy.save()

    # Final evaluation
    ret, eplen = evaluate_policy(env_fn(), policy)
    logger.add_scalar("Test/Return", ret, total_updates)
    logger.add_scalar("Test/Eplen", eplen, total_updates)

    # Final Policy Save
    if args.save_models:
        policy.save()