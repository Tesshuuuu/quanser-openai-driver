import matplotlib.pyplot as plt

from scipy.integrate import odeint
import numpy as np
import math

from gym_brt.quanser import QubeSimulator
from gym_brt.quanser.qube_interfaces import (
    forward_model_ode,
    forward_model_euler,
)

from gym_brt.control import (
    zero_policy,
    constant_policy,
    square_wave_policy,
    pd_control_policy,
    flip_and_hold_policy,
    square_wave_flip_and_hold_policy,
)

def run_sim(forward_model, init_state, policy, nsteps, frequency, integration_steps):
    qube = QubeSimulator(forward_model, init_state, frequency, integration_steps)
    