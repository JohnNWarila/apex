"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore, Style

import time

import numpy as np
import os
import sys

import ray

from rl.envs import WrapEnv

from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.policies.critic import FF_V, LSTM_V
from rl.envs.normalize import get_normalization_params, PreNormalizer

class PPOBuffer:
    """
    A buffer for storing trajectory data and calculating returns for the policy
    and critic updates.

    This container is intentionally not optimized w.r.t. to memory allocation
    speed because such allocation is almost never a bottleneck for policy 
    gradient. 
    
    On the other hand, experience buffers are a frequent source of
    off-by-one errors and other bugs in policy gradient implementations, so
    this code is optimized for clarity and readability, at the expense of being
    (very) marginally slower than some other implementations. 

    (Premature optimization is the root of all evil).
    """
    def __init__(self, gamma=0.99, lam=0.95, use_gae=False):
        self.states  = []
        self.actions = []
        self.rewards = []
        self.values  = []
        self.returns = []

        self.ep_returns = [] # for logging
        self.ep_lens    = []

        self.gamma, self.lam = gamma, lam

        self.ptr, self.path_idx = 0, 0
        self.traj_idx = [0]
    
    def __len__(self):
        return len(self.states)

    def storage_size(self):
        return len(self.states)

    def store(self, state, action, reward, value):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        # TODO: make sure these dimensions really make sense
        self.states  += [state.squeeze(0)]
        self.actions += [action.squeeze(0)]
        self.rewards += [reward.squeeze(0)]
        self.values  += [value.squeeze(0)]

        self.ptr += 1
    
    def finish_path(self, last_val=None):
        self.traj_idx += [self.ptr]

        if last_val is None:
            last_val = np.zeros(shape=(1,))

        path = slice(self.path_idx, self.ptr)
        rewards = self.rewards[path]

        returns = []

        R = last_val.squeeze(0).copy() # Avoid copy?
        for reward in reversed(rewards):
            R = self.gamma * R + reward
            returns.insert(0, R) # TODO: self.returns.insert(self.path_idx, R) ? 
                                 # also technically O(k^2), may be worth just reversing list
                                 # BUG? This is adding copies of R by reference (?)

        self.returns += returns

        self.ep_returns += [np.sum(rewards)]
        self.ep_lens    += [len(rewards)]

        self.path_idx = self.ptr
    
    def get(self):
        return(
            self.states,
            self.actions,
            self.returns,
            self.values
        )

class PPO:
    def __init__(self, args):
        self.env_name       = args['env_name']
        self.gamma          = args['gamma']
        self.lam            = args['lam']
        self.lr             = args['lr']
        self.eps            = args['eps']
        self.entropy_coeff  = args['entropy_coeff']
        self.clip           = args['clip']
        self.minibatch_size = args['minibatch_size']
        self.epochs         = args['epochs']
        self.num_steps      = args['num_steps']
        self.max_traj_len   = args['max_traj_len']
        self.name           = args['policy_name']
        self.use_gae        = args['use_gae']
        self.n_proc         = args['num_procs']
        self.grad_clip      = args['max_grad_norm']
        self.recurrent      = args['recurrent']

        self.max_return = 0
        self.total_steps = 0
        self.highest_reward = -1
        self.limit_cores = 0

        if args['redis_address'] is not None:
            ray.init(redis_address=args['redis_address'])
        else:
            ray.init()

    def save(self, policy, critic):
        # save_path = os.path.join("./trained_models", "ppo")
        # try:
        #     os.makedirs(save_path)
        # except OSError:
        #     pass
        filetype = ".pt" # pytorch model
        torch.save(policy, os.path.join("./trained_models", self.name + "_actor"+filetype))
        torch.save(critic, os.path.join("./trained_models", self.name + "_critic"+filetype))

    @ray.remote
    @torch.no_grad()
    def sample(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False):
        """
        Sample at least min_steps number of total timesteps, truncating 
        trajectories only if they exceed max_traj_len number of timesteps
        """
        torch.set_num_threads(1) # By default, PyTorch will use multiple cores to speed up operations.
                                 # This can cause issues when Ray also uses multiple cores, especially on machines
                                 # with a lot of CPUs. I observed a significant speedup when limiting PyTorch 
                                 # to a single core - I think it basically stopped ray workers from stepping on each
                                 # other's toes.

        env = WrapEnv(env_fn) # TODO

        memory = PPOBuffer(self.gamma, self.lam)

        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            value = 0
            traj_len = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()

            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=False)
                value = critic(state)

                next_state, reward, done, _ = env.step(action.numpy())

                memory.store(state.numpy(), action.numpy(), reward, value.numpy())

                state = torch.Tensor(next_state)

                traj_len += 1
                num_steps += 1

            value = critic(state)
            memory.finish_path(last_val=(not done) * value.numpy())

        return memory

    def sample_parallel(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False):
        worker = self.sample
        args = (self, env_fn, policy, critic, min_steps, max_traj_len, deterministic)

        # Don't don't bother launching another process for single thread
        if self.n_proc > 1:
            result = ray.get([worker.remote(*args) for _ in range(self.n_proc)])
            real_proc = self.n_proc
            if self.limit_cores:
                real_proc = 50 - 16*int(np.log2(60 / env_fn().simrate))
                print("limit cores active, using {} cores".format(real_proc))
                args = (self, env_fn, policy, critic, min_steps*self.n_proc // real_proc, max_traj_len, deterministic)
            result = ray.get([worker.remote(*args) for _ in range(real_proc)])
            # result_ids = [worker.remote(*args) for _ in range(real_proc)]
            # result = ray.get(result_ids)
        else:
            result = [worker._function(*args)]

        # O(n)
        def merge(buffers):
            merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:
                offset = len(merged)
                merged.states  += buf.states
                merged.actions += buf.actions
                merged.rewards += buf.rewards
                merged.values  += buf.values
                merged.returns += buf.returns

                merged.ep_returns += buf.ep_returns
                merged.ep_lens    += buf.ep_lens

                merged.traj_idx += [offset + i for i in buf.traj_idx[1:]]
                merged.ptr      += buf.ptr

            return merged
        total_buf = merge(result)
        if len(total_buf) > min_steps*self.n_proc * 1.5:
            self.limit_cores = 1
        return total_buf

    def update_policy(self, obs_batch, action_batch, return_batch, advantage_batch, mask, env, mirror_observation=None, mirror_action=None):
        policy = self.policy
        critic = self.critic
        old_policy = self.old_policy

        values = critic(obs_batch)
        pdf = policy.distribution(obs_batch)

        # TODO, move this outside loop?
        with torch.no_grad():
            old_pdf = old_policy.distribution(obs_batch)
            old_log_probs = old_pdf.log_prob(action_batch).sum(-1, keepdim=True)
        
        log_probs = pdf.log_prob(action_batch).sum(-1, keepdim=True)
        
        ratio = (log_probs - old_log_probs).exp()

        cpi_loss = ratio * advantage_batch * mask
        clip_loss = ratio.clamp(1.0 - self.clip, 1.0 + self.clip) * advantage_batch * mask
        actor_loss = -torch.min(cpi_loss, clip_loss).mean()

        critic_loss = 0.5 * ((return_batch - values) * mask).pow(2).mean()

        entropy_penalty = -(self.entropy_coeff * pdf.entropy() * mask).mean()

        # Mirror Symmetry Loss
        if mirror_observation is not None and mirror_action is not None:
          deterministic_actions = policy(obs_batch)
          if env.clock_based:
              mir_obs = mirror_observation(obs_batch, env.clock_inds)
              mirror_actions = policy(mir_obs)
          else: 
              mirror_actions = policy(mirror_observation(obs_batch))
          mirror_actions = mirror_action(mirror_actions)
          mirror_loss = 4 * (deterministic_actions - mirror_actions).pow(2).mean()
        else:
          mirror_loss = 0 

        self.actor_optimizer.zero_grad()
        (actor_loss + mirror_loss + entropy_penalty).backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from 
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(policy.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Clip the gradient norm to prevent "unlucky" minibatches from 
        # causing pathological updates
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        with torch.no_grad():
          kl = kl_divergence(pdf, old_pdf)

        return actor_loss.item(), pdf.entropy().mean().item(), critic_loss.item(), ratio.mean().item(), kl.mean().item()

    def train(self,
              env_fn,
              policy,
              critic,
              n_itr,
              logger=None):

        self.old_policy = deepcopy(policy)
        self.policy = policy
        self.critic = critic

        self.actor_optimizer = optim.Adam(policy.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=self.lr, eps=self.eps)

        start_time = time.time()
        opt_time = np.zeros(n_itr)
        samp_time = np.zeros(n_itr)
        eval_time = np.zeros(n_itr)

        env = env_fn()
        obs_mirr, act_mirr = None, None
        if hasattr(env, 'mirror_observation'):
            if env.clock_based:
                obs_mirr = env.mirror_clock_observation
            else:
                obs_mirr = env.mirror_observation

        if hasattr(env, 'mirror_action'):
          act_mirr = env.mirror_action

        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            sample_start = time.time()
            batch = self.sample_parallel(env_fn, self.policy, self.critic, self.num_steps, self.max_traj_len)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            samp_time[itr] = time.time() - sample_start
            print("sample time elapsed: {:.2f} s".format(samp_time[itr]))

            observations, actions, returns, values = map(torch.Tensor, batch.get())

            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

            minibatch_size = self.minibatch_size or advantages.numel()

            print("timesteps in batch: %i" % advantages.numel())
            self.total_steps += advantages.numel()

            self.old_policy.load_state_dict(policy.state_dict())

            optimizer_start = time.time()
            
            for _ in range(self.epochs):
                losses = []
                entropies = []
                kls = []
                if self.recurrent:
                    random_indices = SubsetRandomSampler(range(len(batch.traj_idx)-1))
                else:
                    random_indices = SubsetRandomSampler(range(advantages.numel()))

                sampler = BatchSampler(random_indices, minibatch_size, drop_last=True)
                for indices in sampler:
                    if self.recurrent:
                        obs_batch       = [observations[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        action_batch    = [actions[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        return_batch    = [returns[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        advantage_batch = [advantages[batch.traj_idx[i]:batch.traj_idx[i+1]] for i in indices]
                        mask            = [torch.ones_like(r) for r in return_batch]

                        obs_batch       = pad_sequence(obs_batch, batch_first=False)
                        action_batch    = pad_sequence(action_batch, batch_first=False)
                        return_batch    = pad_sequence(return_batch, batch_first=False)
                        advantage_batch = pad_sequence(advantage_batch, batch_first=False)
                        mask            = pad_sequence(mask, batch_first=False)
                    else:
                        obs_batch       = observations[indices]
                        action_batch    = actions[indices]
                        return_batch    = returns[indices]
                        advantage_batch = advantages[indices]
                        mask            = 1

                    scalars = self.update_policy(obs_batch, action_batch, return_batch, advantage_batch, mask, env, mirror_observation=obs_mirr, mirror_action=act_mirr)
                    actor_loss, entropy, critic_loss, ratio, kl = scalars

                    entropies.append(entropy)
                    kls.append(kl)
                    losses.append([actor_loss, entropy, critic_loss, ratio, kl])
                    
                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g"%x for x in np.mean(losses, axis=0)]))

                # Early stopping 
                if np.mean(kl) > 0.02:
                    print("Max kl reached, stopping optimization early.")
                    break

            opt_time[itr] = time.time() - optimizer_start
            print("optimizer time elapsed: {:.2f} s".format(opt_time[itr]))        

            if logger is not None:
                evaluate_start = time.time()
                eval_proc = min(self.n_proc, 24)
                test = self.sample_parallel(env_fn, self.policy, self.critic, 800 // eval_proc, self.max_traj_len, deterministic=True)
                eval_time[itr] = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time[itr]))                

                avg_eval_reward = np.mean(test.ep_returns)
                print("avg eval reward: {:.2f}".format(avg_eval_reward))
                avg_batch_reward = np.mean(batch.ep_returns)
                avg_ep_len = np.mean(batch.ep_lens)
                pdf = policy.distribution(observations)

                entropy = np.mean(entropies)
                kl = np.mean(kls)

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                # logger.add_scalar("Test/Return", avg_eval_reward, itr)
                # logger.add_scalar("Train/Return", np.mean(batch.ep_returns), itr)
                # logger.add_scalar("Train/Mean Eplen", np.mean(batch.ep_lens), itr)
                # logger.add_scalar("Train/Mean KL Div", kl, itr)
                # logger.add_scalar("Train/Mean Entropy", entropy, itr)
                # logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

                logger.add_scalar("Data/Return (test)", avg_eval_reward, itr)
                logger.add_scalar("Data/Return (batch)", avg_batch_reward, itr)
                logger.add_scalar("Data/Mean Eplen", avg_ep_len, itr)

                logger.add_scalar("Misc/Mean KL Div", kl, itr)
                logger.add_scalar("Misc/Mean Entropy", entropy, itr)
                logger.add_scalar("Misc/Critic Loss", critic_loss, itr)
                logger.add_scalar("Misc/Actor Loss", actor_loss, itr)

                logger.add_scalar("Misc/Sample Times", samp_time[itr], itr)
                logger.add_scalar("Misc/Optimize Times", opt_time[itr], itr)
                logger.add_scalar("Misc/Evaluation Times", eval_time[itr], itr)

                for i in range(pdf.loc.shape[1]): # go thru al actions
                    logger.add_histogram("Action Dist/action_"+str(i), pdf.loc[:,i], itr)

            # TODO: add option for how often to save model
            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                # torch.save(policy, os.path.join(logger.dir, '_actor.pt'))
                self.save(policy, critic)

def run_experiment(args):
    from apex import env_factory, create_logger

    torch.set_num_threads(1)

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror, speed=args.speed, clock_based=args.clock_based)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        policy = torch.load(args.previous)
        print("loaded model from {}".format(args.previous))
    else:
        if args.recurrent:
            policy = Gaussian_LSTM_Actor(obs_dim, action_dim, fixed_std=np.exp(-2), env_name=args.env_name)
            critic = LSTM_V(obs_dim)
        else:
            policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=np.exp(-2), env_name=args.env_name)
            critic = FF_V(obs_dim)

        with torch.no_grad():
            policy.obs_mean, policy.obs_std = map(torch.Tensor, get_normalization_params(iter=args.input_norm_steps, noise_std=1, policy=policy, env_fn=env_fn))
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    algo = PPO(args=vars(args))

    # create a tensorboard logging object
    # logger = create_logger(args)
    log_path = args.logdir + args.policy_name + "/"
    logger = SummaryWriter(log_path, flush_secs=0.1)
    print(Fore.GREEN + Style.BRIGHT + "Logging data using TensorBoard to {}".format(log_path + Style.RESET_ALL))

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print("\tenv:            {}".format(args.env_name))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print("\tseed:           {}".format(args.seed))
    print("\tmirror:         {}".format(args.mirror))
    print("\tnum procs:      {}".format(args.num_procs))
    print("\tlr:             {}".format(args.lr))
    print("\teps:            {}".format(args.eps))
    print("\tlam:            {}".format(args.lam))
    print("\tgamma:          {}".format(args.gamma))
    print("\tentropy coeff:  {}".format(args.entropy_coeff))
    print("\tclip:           {}".format(args.clip))
    print("\tminibatch size: {}".format(args.minibatch_size))
    print("\tepochs:         {}".format(args.epochs))
    print("\tnum steps:      {}".format(args.num_steps))
    print("\tuse gae:        {}".format(args.use_gae))
    print("\tmax grad norm:  {}".format(args.max_grad_norm))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(env_fn, policy, critic, args.n_itr, logger=logger)

