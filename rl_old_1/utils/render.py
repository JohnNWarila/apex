import torch
import numpy as np
from torch.autograd import Variable
import time
import sys, tty, termios, select

from .quaternion_function import quaternion_product
from .quaternion_function import inverse_quaternion
from .quaternion_function import euler2quat
from .quaternion_function import rotate_by_quaternion


@torch.no_grad()
def renderpolicy(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset())
    while True:
        _, action = policy.act(state, deterministic)

        state, reward, done, _ = env.step(action.data.numpy())
        done=False
        if done:
            state = env.reset()

        state = torch.Tensor(state)

        env.render()

        time.sleep(dt / speedup)

def renderloop(env, policy, deterministic=False, speedup=1):
    while True:
        renderpolicy(env, policy, deterministic, speedup)


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    
@torch.no_grad()
def renderpolicy_speedinput(env, policy, deterministic=False, speedup=1, dt=0.05):
    state = torch.Tensor(env.reset_for_test())
    env.speed = 0
    env.side_speed = 0
    env.phase_add = 1
    orient_add = 0

    # Check if using StateEst or not
    if env.observation_space.shape[0] >= 48:
        is_stateest = True
    else:
        is_stateest = False

    render_state = env.render()
    old_settings = termios.tcgetattr(sys.stdin)
    print("render_state:", render_state)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while render_state:
            if isData():
                c = sys.stdin.read(1)
                if c == 'w':
                    env.speed += .1
                    print("Increasing speed to: ", env.speed)
                elif c == 's':
                    env.speed -= .1
                    print("Decreasing speed to: ", env.speed)
                elif c == 'a':
                    env.side_speed += .1
                    print("Increasing side speed to: ", env.side_speed)
                elif c == 'd':
                    env.side_speed -= .1
                    print("Decreasing side speed to: ", env.side_speed)
                elif c == 'j':
                    env.phase_add += .1
                    print("Increasing frequency to: ", env.phase_add)
                elif c == 'h':
                    env.phase_add -= .1
                    print("Decreasing frequency to: ", env.phase_add)
                elif c == 'l':
                    orient_add += .1
                    print("Increasing orient_add to: ", orient_add)
                elif c == 'k':
                    orient_add -= .1
                    print("Decreasing orient_add to: ", orient_add)
                elif c == 'p':
                    print("Applying force")
                    push = 200
                    push_dir = 0
                    force_arr = np.zeros(6)
                    force_arr[push_dir] = push
                    env.sim.apply_force(force_arr)
                else:
                    pass
            if (not env.vis.ispaused()):
                # Update orientation
                quaternion = euler2quat(z=orient_add, y=0, x=0)
                iquaternion = inverse_quaternion(quaternion)
                if is_stateest:
                    curr_orient = state[1:5]
                    curr_transvel = state[14:17]
                else:
                    curr_orient = state[2:6]
                    curr_transvel = state[20:23]
                new_orient = quaternion_product(iquaternion, curr_orient)
                if new_orient[0] < 0:
                    new_orient = -new_orient
                new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                if is_stateest:
                    state[1:5] = torch.FloatTensor(new_orient)
                    state[14:17] = torch.FloatTensor(new_translationalVelocity)
                    state[0] = 1      # For use with StateEst. Replicate hack that height is always set to one on hardware.
                else:
                    state[2:6] = torch.FloatTensor(new_orient)
                    state[20:23] = torch.FloatTensor(new_translationalVelocity)
               
                # Get action
                action = policy.act(state, deterministic)
                if deterministic:
                    action = action.data.numpy()
                else:
                    action = action.data[0].numpy()

                state, reward, done, _ = env.step(action)
                foot_pos = np.zeros(6)
                env.sim.foot_pos(foot_pos)
                foot_forces = env.sim.get_foot_forces()
                # print("Foot force norm: ", foot_forces[0])
                # print("foot distance: ", np.linalg.norm(foot_pos[0:3]-foot_pos[3:6]))
                # print("speed: ", env.sim.qvel()[0])
                # print("desired speed: ", env.speed)
                # print("pelvis accel: ", np.linalg.norm(env.cassie_state.pelvis.translationalAcceleration))

                # if done:
                #     state = env.reset()

                state = torch.Tensor(state)

            render_state = env.render()
            time.sleep(dt / speedup)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)