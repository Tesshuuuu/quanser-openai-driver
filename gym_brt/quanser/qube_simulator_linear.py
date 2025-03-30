import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from jax.scipy.linalg import expm
from jax import jit

# Motor
Rm = 8.4  # Resistance
kt = 0.042  # Current-torque (N-m/A)
km = 0.042  # Back-emf constant (V-s/rad)

# Rotary Arm
mr = 0.095  # Mass (kg)
Lr = 0.085  # Total length (m)
Jr = mr * Lr ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dr = 0.00027  # Equivalent viscous damping coefficient (N-m-s/rad)

# Pendulum Link
# mp = 0.024  # Mass (kg)
# Lp = 0.129  # Total length (m)
# Jp = mp * Lp ** 2 / 12  # Moment of inertia about pivot (kg-m^2)
Dp = 0.00005  # Equivalent viscous damping coefficient (N-m-s/rad)

g = 9.81  # Gravity constant

@jit
def diff_forward_model_ode(state, action, mass, length):
    mp = mass
    Lp = length
    Jp = mp * Lp ** 2 / 12
    theta, alpha, theta_dot, alpha_dot = state
    Vm = action
    tau = (km * (Vm - km * theta_dot)) / Rm  # torque
    # fmt: off
    # From Rotary Pendulum Workbook
    theta_dot_dot = (-Lp*Lr*mp*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*jnp.sin(2.0*alpha) + 4.0*Lp*g*mp*jnp.sin(alpha))*jnp.cos(alpha) + (4.0*Jp + Lp**2*mp)*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*jnp.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*jnp.sin(alpha) - 4.0*tau))/(4.0*Lp**2*Lr**2*mp**2*jnp.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*jnp.sin(alpha)**2 + 4.0*Lr**2*mp))
    alpha_dot_dot = (2.0*Lp*Lr*mp*(4.0*Dr*theta_dot + Lp**2*alpha_dot*mp*theta_dot*jnp.sin(2.0*alpha) + 2.0*Lp*Lr*alpha_dot**2*mp*jnp.sin(alpha) - 4.0*tau)*jnp.cos(alpha) - 0.5*(4.0*Jr + Lp**2*mp*jnp.sin(alpha)**2 + 4.0*Lr**2*mp)*(-8.0*Dp*alpha_dot + Lp**2*mp*theta_dot**2*jnp.sin(2.0*alpha) + 4.0*Lp*g*mp*jnp.sin(alpha)))/(4.0*Lp**2*Lr**2*mp**2*jnp.cos(alpha)**2 - (4.0*Jp + Lp**2*mp)*(4.0*Jr + Lp**2*mp*jnp.sin(alpha)**2 + 4.0*Lr**2*mp))
    # fmt: on

    return jnp.array([theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot])

@jit
def forward_model_ode(state, action, dt, mass, length):
    t = jnp.linspace(0.0, dt, 2)  
    theta_dot, alpha_dot, theta_dot_dot, alpha_dot_dot = diff_forward_model_ode(state, action, mass, length)

    next_state = odeint(lambda state, t: diff_forward_model_ode(state, action, mass, length), state, t)[-1]
    theta, alpha, theta_dot, alpha_dot = next_state

    theta = ((theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    alpha = ((alpha + jnp.pi) % (2 * jnp.pi)) - jnp.pi

    return jnp.array([theta, alpha, theta_dot, alpha_dot])


@jit
def dynamics(state, action, dt, mass, length):
    return forward_model_ode(state, action, dt, mass, length)

@jit
def linearize_dynamics(state, action, dt, mass, length):
    Al = jax.jacobian(dynamics, argnums=0)(state, action, dt, mass, length)
    Bl = jax.jacobian(dynamics, argnums=1)(state, action, dt, mass, length)
    return Al, Bl

@jit
def discretize_linearize_dynamics(state, action, dt, mass, length):
    A, B = linearize_dynamics(state, action, dt, mass, length)
    n = A.shape[0]
    B = B.reshape(-1, 1)
    m = B.shape[1]
    
    M = jnp.block([
        [A, B],
        [jnp.zeros((m, n + m))]
    ])
    M_exp = expm(M * dt)
    Ad = M_exp[:n, :n]
    Bd = M_exp[:n, n:n+m]
    return Ad, Bd

if __name__ == "__main__":
    state0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    action0 = 1.0
    dt = 0.01
    mass = 0.024
    length = 0.129
    Ad, Bd = discretize_linearize_dynamics(state0, action0, dt, mass, length)
    print("Discretized Linear A:\n", Ad)
    print("Discretized Linear B:\n", Bd)
    