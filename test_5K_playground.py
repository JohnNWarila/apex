import torch
import hashlib, os, pickle
from collections import OrderedDict

from cassie.quaternion_function import *


def eval_policy(policy, args, render=True):

    import tty
    import termios
    import select
    import numpy as np
    from cassie import CassieEnv, CassiePlayground, CassieStandingEnv

    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    max_traj_len = args.traj_len
    visualize = True if render else False

    if args.env_name == "Cassie-v0":
        env = CassieEnv(traj=args.traj, state_est=args.state_est, dynamics_randomization=args.dyn_random, clock_based=args.clock_based, reward=args.reward, history=args.history)
    elif args.env_name == "CassiePlayground-v0":
        env = CassiePlayground(traj=args.traj, state_est=args.state_est, dynamics_randomization=args.dyn_random, clock_based=args.clock_based, reward=args.reward, history=args.history)
    else:
        env = CassieStandingEnv(state_est=args.state_est)
    

    old_settings = termios.tcgetattr(sys.stdin)

    orient_add = 0

    try:
        tty.setcbreak(sys.stdin.fileno())

        state = env.reset_for_test()
        done = False
        timesteps = 0
        eval_reward = 0
        speed = 0.0

        while True:
        
            if isData():
                c = sys.stdin.read(1)
                if c == 'w':
                    speed += 0.1
                elif c == 's':
                    speed -= 0.1
                elif c == 'l':
                    orient_add += .1
                    print("Increasing orient_add to: ", orient_add)
                elif c == 'k':
                    orient_add -= .1
                    print("Decreasing orient_add to: ", orient_add)
                elif c == 'p':
                    push = 100
                    push_dir = 2
                    force_arr = np.zeros(6)
                    force_arr[push_dir] = push
                    env.sim.apply_force(force_arr)

                env.update_speed(speed)
                print("speed: ", env.speed)
            
            # Update Orientation
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)

            if env.state_est:
                curr_orient = state[1:5]
                curr_transvel = state[14:17]
            else:
                curr_orient = state[2:6]
                curr_transvel = state[20:23]
            
            new_orient = quaternion_product(iquaternion, curr_orient)

            if new_orient[0] < 0:
                new_orient = -new_orient

            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            
            if env.state_est:
                state[1:5] = torch.FloatTensor(new_orient)
                state[14:17] = torch.FloatTensor(new_translationalVelocity)
                # state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
            else:
                state[2:6] = torch.FloatTensor(new_orient)
                state[20:23] = torch.FloatTensor(new_translationalVelocity)

            if hasattr(env, 'simrate'):
                start = time.time()
                
            action = policy.forward(torch.Tensor(state), deterministic=True).detach().numpy()
            state, reward, done, _ = env.step(action)
            if visualize:
                env.render()
            eval_reward += reward
            timesteps += 1

            if hasattr(env, 'simrate'):
                # assume 30hz (hack)
                end = time.time()
                delaytime = max(0, 1000 / 30000 - (end-start))
                time.sleep(delaytime)

        print("Eval reward: ", eval_reward)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    import sys, argparse, time
    parser = argparse.ArgumentParser()

    """
        General arguments for configuring the environment
    """
    parser.add_argument("--traj", default="walking", type=str, help="reference trajectory to use. options are 'aslip', 'walking', 'stepping'")
    parser.add_argument("--clock_based", default=True, action='store_true')
    parser.add_argument("--state_est", default=True, action='store_true')
    parser.add_argument("--dyn_random", default=False, action='store_true')
    parser.add_argument("--no_delta", default=True, action='store_true')
    parser.add_argument("--reward", default="iros_paper", type=str)
    parser.add_argument("--mirror", default=False, action='store_true')                     # mirror actions or not

    parser.add_argument("--path", type=str, default="./trained_models/5k/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2_actor.pt", help="path to folder containing policy and run details")
    parser.add_argument("--env_name", default="Cassie-v0", type=str)
    parser.add_argument("--traj_len", default=400, type=str)
    parser.add_argument("--history", default=0, type=int)                                   # number of previous states to use as input
    args = parser.parse_args()

    # run_args = pickle.load(open(args.path + "experiment.pkl", "rb"))

    policy = torch.load(args.path)
    policy.eval()

    eval_policy(policy, args, render=True)
    