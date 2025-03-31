from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from gym_brt.envs import (
    QubeSwingupEnv,
    QubeSwingupSparseEnv,
    QubeSwingupFollowEnv,
    QubeSwingupFollowSparseEnv,
    QubeBalanceEnv,
    QubeBalanceSparseEnv,
    QubeBalanceFollowEnv,
    QubeBalanceFollowSparseEnv,
    QubeDampenEnv,
    QubeDampenSparseEnv,
    QubeDampenFollowEnv,
    QubeDampenFollowSparseEnv,
    QubeRotorEnv,
    QubeRotorFollowEnv,
    QubeBalanceFollowSineWaveEnv,
    QubeSwingupFollowSineWaveEnv,
    QubeRotorFollowSineWaveEnv,
    QubeDampenFollowSineWaveEnv,
)

from gym_brt.control import (
    zero_policy,
    constant_policy,
    random_policy,
    square_wave_policy,
    energy_control_policy,
    pd_control_policy,
    flip_and_hold_policy,
    square_wave_flip_and_hold_policy,
    dampen_policy,
    pd_tracking_control_policy,
)


def print_info(state_info, action, reward):
    theta = state_info["theta"]
    alpha = state_info["alpha"]
    theta_dot = state_info["theta_dot"]
    alpha_dot = state_info["alpha_dot"]
    print(
        "State: theta={:06.3f}, alpha={:06.3f}, theta_dot={:06.3f}, alpha_dot={:06.3f}".format(
            theta, alpha, theta_dot, alpha_dot
        )
    )
    print("Action={}, Reward={}".format(action, reward))


def test_env(
    env_name,
    policy,
    num_episodes=10,
    num_steps=250,
    frequency=250,
    state_keys=None,
    verbose=False,
    use_simulator=False,
    render=False,
    kp_theta=None,
    kp_alpha=None,
    kd_theta=None,
    kd_alpha=None,
):
    theta_lst = []
    alpha_lst = []
    with env_name(use_simulator=use_simulator, frequency=frequency) as env:
        for episode in range(num_episodes):
            state = env.reset()
            state, reward, done, info = env.step(np.array([0], dtype=np.float64))
            for step in range(num_steps):
                action = policy(state, step=step, frequency=frequency, kp_theta=kp_theta, kp_alpha=kp_alpha, kd_theta=kd_theta, kd_alpha=kd_alpha)
                state, reward, done, info = env.step(action)
                theta_lst.append(info["theta"])
                alpha_lst.append(info["alpha"])
                if done:
                    print("Episode {} finished".format(episode))
                    break
                if verbose:
                    print_info(info, action, reward)
                if render:
                    env.render()

    return theta_lst, alpha_lst

def get_gains_from_keyboard():
    print("\nEnter controller gains:")
    kp_theta = float(input("Enter kp_theta (default=-2.0): ") or "-2.0")
    kp_alpha = float(input("Enter kp_alpha (default=35.0): ") or "35.0")
    kd_theta = float(input("Enter kd_theta (default=-1.5): ") or "-1.5")
    kd_alpha = float(input("Enter kd_alpha (default=3.0): ") or "3.0")
    
    print("\nUsing gains:")
    print(f"kp_theta: {kp_theta}")
    print(f"kp_alpha: {kp_alpha}")
    print(f"kd_theta: {kd_theta}")
    print(f"kd_alpha: {kd_alpha}")
    
    return kp_theta, kp_alpha, kd_theta, kd_alpha

def main():
    envs = {
        "QubeSwingupEnv": QubeSwingupEnv,
        "QubeSwingupSparseEnv": QubeSwingupSparseEnv,
        "QubeSwingupFollowEnv": QubeSwingupFollowEnv,
        "QubeSwingupFollowSparseEnv": QubeSwingupFollowSparseEnv,
        "QubeBalanceEnv": QubeBalanceEnv,
        "QubeBalanceSparseEnv": QubeBalanceSparseEnv,
        "QubeBalanceFollowEnv": QubeBalanceFollowEnv,
        "QubeBalanceFollowSparseEnv": QubeBalanceFollowSparseEnv,
        "QubeDampenEnv": QubeDampenEnv,
        "QubeDampenSparseEnv": QubeDampenSparseEnv,
        "QubeDampenFollowEnv": QubeDampenFollowEnv,
        "QubeDampenFollowSparseEnv": QubeDampenFollowSparseEnv,
        "QubeRotorEnv": QubeRotorEnv,
        "QubeRotorFollowEnv": QubeRotorFollowEnv,
        "QubeBalanceFollowSineWaveEnv": QubeBalanceFollowSineWaveEnv,
        "QubeSwingupFollowSineWaveEnv": QubeSwingupFollowSineWaveEnv,
        "QubeRotorFollowSineWaveEnv": QubeRotorFollowSineWaveEnv,
        "QubeDampenFollowSineWaveEnv": QubeDampenFollowSineWaveEnv,
    }
    policies = {
        "none": zero_policy,
        "zero": zero_policy,
        "const": constant_policy,
        "rand": random_policy,
        "random": random_policy,
        "sw": square_wave_policy,
        "energy": energy_control_policy,
        "pd": pd_control_policy,
        "hold": pd_control_policy,
        "flip": flip_and_hold_policy,
        "sw-hold": square_wave_flip_and_hold_policy,
        "damp": dampen_policy,
        "track": pd_tracking_control_policy,
    }

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        default="QubeSwingupEnv",
        choices=list(envs.keys()),
        help="Enviroment to run.",
    )
    parser.add_argument(
        "-c",
        "--controller",
        default="flip",
        choices=list(policies.keys()),
        help="Select what type of action to take.",
    )
    parser.add_argument(
        "-ne",
        "--num-episodes",
        default="1",
        type=int,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "-ns",
        "--num-steps",
        default="10000",
        type=int,
        help="Number of step to run per episode.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        "--sample-frequency",
        default="250",
        type=float,
        help="The frequency of samples on the Quanser hardware.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-s", "--use_simulator", action="store_true")
    args = parser.parse_args()

    # Get gains from keyboard input
    # kp_theta, kp_alpha, kd_theta, kd_alpha = get_gains_from_keyboard()
    


    # Update args with the keyboard inputs
    DR = True
    if DR:
        args.kp_theta = -0.6562398
        args.kp_alpha = 39.63346
        args.kd_theta = -1.0030441
        args.kd_alpha = 5.8323298
    else:
        # kp_theta, kp_alpha, kd_theta, kd_alpha = get_gains_from_keyboard()
        args.kp_theta = -0.36665
        args.kp_alpha = 27.454126
        args.kd_theta = -0.509547
        args.kd_alpha = 2.4390068

    print("Testing Env:  {}".format(args.env))
    print("Controller:   {}".format(args.controller))
    print("{} steps over {} episodes".format(args.num_steps, args.num_episodes))
    print("Samples freq: {}".format(args.frequency))
    theta_lst, alpha_lst = test_env(
        envs[args.env],
        policies[args.controller],
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        frequency=args.frequency,
        verbose=args.verbose,
        use_simulator=False,
        render=False,
        kp_theta=args.kp_theta,
        kp_alpha=args.kp_alpha,
        kd_theta=args.kd_theta,
        kd_alpha=args.kd_alpha,
    )
    plt.figure(figsize=(10, 5))
    plt.plot(theta_lst, label="theta")
    plt.plot(alpha_lst, label="alpha")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
