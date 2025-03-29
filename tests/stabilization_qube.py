import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit, vmap, random

import pickle
import numpy as np
import scipy
import scipy.linalg as la

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from gym_brt.quanser.qube_simulator_linear import discretize_linearize_dynamics

state0 = jnp.array([0.0, 0.0, 0.0, 0.0])
action0 = 0.1
dt = 0.01

m_min, m_max = 0.024, 0.024
ell_min, ell_max = 0.129, 0.129

dx = 4
du = 1
Q = jnp.eye(dx)
R = jnp.eye(du)
Sigma_w = jnp.eye(dx)

'''
  Define the cost function and its gradient
'''
@jit
def spec_rad(As):
  return jnp.max(jnp.abs(jnp.linalg.eigvals(As)), axis=-1)

@jit
def dlyap_direct(A, Q):
    def true_fn(_):
      lhs = jnp.kron(A, jnp.conj(A))
      lhs = jnp.eye(lhs.shape[0]) - lhs
      x = jnp.linalg.solve(lhs, Q.flatten())
      return jnp.reshape(x, Q.shape)
    
    def false_fn(_):
      return 1000000*jnp.eye(Q.shape[0])
    
    eigvals_product = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    x = jax.lax.cond(eigvals_product < 1, true_fn, false_fn, None)
    return x

def cost(K, As, Bs):
  ABK = As + jnp.einsum('...ij, jk -> ...ik', Bs, K)
  spec_rads = spec_rad(ABK)

  def compute_cost_fn(A, B):
    def true_fn(_):
      P = dlyap_direct((A+B@K).T, Q + K.T@R@K)
      return jnp.trace(P)
    
    def false_fn(_):
      return jnp.array(100000, dtype=jnp.float32)
    
    return jax.lax.cond(
      spec_rad(A+B@K) < 1,
      true_fn,
      false_fn,
      None
    )
    
  costs = vmap(compute_cost_fn, (0, 0))(As, Bs)
  return jnp.mean(costs)

def grad(K, As, Bs):
    def single_grad(A, B):
        P = dlyap_direct((A+B@K).T, Q + K.T@R@K)
        S = dlyap_direct(A+B@K, jnp.eye(dx))
        E = (R + B.T@P@B)@K + B.T@P@A
        return 2*E@S
    grads = vmap(single_grad, (0, 0))(As, Bs)
    return jnp.mean(grads, axis=0)

def grad_descent(K, As, Bs, alpha, n_iterations):
    def body_fn(i, K):
        grad_k = grad(K, As, Bs)  # Use grad directly instead of avg_grad
        K = K - alpha*grad_k
        # ensure all elements of K are identical
        K = jnp.full(K.shape, jnp.mean(K))
        return K
    
    return jax.lax.fori_loop(0, n_iterations, body_fn, K)

# progressive discounting: find gamma satisfying 2.5*C_old < C_new < 8*C_old
def update_gamma(gamma, As, Bs, K):
  C_old = cost(K, jnp.sqrt(gamma)*As, jnp.sqrt(gamma)*Bs)

  gamma_lb = gamma
  gamma_ub = 1.0
  # bisection
  while gamma_ub - gamma_lb > 1e-4:
    gamma_mid = (gamma_ub + gamma_lb)/2
    C_new = cost(K, jnp.sqrt(gamma_mid)*As, jnp.sqrt(gamma_mid)*Bs)

    if C_new < 2.5*C_old:
      gamma_lb = gamma_mid
    elif C_new > 4*C_old:
      gamma_ub = gamma_mid
    else:
      gamma_lb = gamma_mid
      break
  return gamma_lb

# n_samples = 100000
# systems = [discretize_linearize_dynamics(state0, action0, dt, mass, length) for mass,length in zip(np.random.uniform(m_min, m_max, n_samples), np.random.uniform(ell_min, ell_max, n_samples))]
# eval_As = jnp.stack([A for A, B in systems])
# eval_Bs = jnp.stack([B for A, B in systems])

alpha = 1e-4
n_iterations = 20
n_itertions_final = 5

n_trials = 1

all_samples_Ks = []
all_samples_SA_cost = []
all_samples_DR_cost = []

for i, n_samples in enumerate([10]):
  all_Ks = []
  all_SA_cost = []
  all_DR_cost = []
  base_key = random.PRNGKey(i+2)
  subkeys = random.split(base_key, n_trials)
  for trial in range(n_trials):
    # generate new subkey for each trial
    subkey = subkeys[trial]
    # print(f'subkey: {subkey}')
    print(f'trial {trial+1} of {n_trials} for n_samples = {n_samples}')
    systems = [discretize_linearize_dynamics(state0, action0, dt, mass, length) for mass,length in zip(random.uniform(subkey, minval = m_min, maxval = m_max, shape = (n_samples,)), random.uniform(subkey, minval = ell_min, maxval = ell_max, shape = (n_samples,)))]
    As = jnp.stack([A for A, B in systems])
    Bs = jnp.stack([B for A, B in systems])
    # print('As: ', As)
    # print('Bs: ', Bs)
    print('As.shape: ', As.shape)
    print('Bs.shape: ', Bs.shape)

    rho = jnp.max(spec_rad(As))
    gamma = min(0.9*rho**(-2), 1)

    K = jnp.zeros((du,dx))

    Ks = []
    SA_cost = []
    # DR_cost = [

    # check if the spectral radius is less than 1 and gamma is less than 1
    while gamma < 0.999:
      K = grad_descent(K, jnp.sqrt(gamma)*As, jnp.sqrt(gamma)*Bs, alpha, n_iterations)
      Ks.append(K)
      SA_cost.append(cost(K, As, Bs))
      # DR_cost.append(cost(K, eval_As, eval_Bs))
    #   print('K: ', K)
      print('spectral radius', max([spec_rad(A+B@K) for (A,B) in zip(As,Bs)]))
    #   print('A + B@K', [A+B@K for (A,B) in zip(As,Bs)])
      print('discounted spectral radius', max([spec_rad((jnp.sqrt(gamma)*A)+(jnp.sqrt(gamma)*B)@K) for (A,B) in zip(As,Bs)]))
      print('iteration: ', len(Ks))
      gamma = update_gamma(gamma, As, Bs, K)
      print('gamma: ', gamma)
    #   print('SA_cost: ', cost(K, As, Bs))
    #   print('DR_cost: ', cost(K, eval_As, eval_Bs))
    for i in range(40):
      K = grad_descent(K, As, Bs, alpha, n_iterations)
      Ks.append(K)
      SA_cost.append(cost(K, As, Bs))
      # DR_cost.append(cost(K, eval_As, eval_Bs))
      print('iteration: ', len(Ks))
    #   print('K: ', K)
    #   print('SA_cost: ', cost(K, As, Bs))
    #   print('DR_cost: ', cost(K, eval_As, eval_Bs))

    print('final controller: ', K)
    all_Ks.append(Ks)
    all_SA_cost.append(SA_cost)
    # all_DR_cost.append(DR_cost)
  
  all_samples_Ks.append(all_Ks)
  all_samples_SA_cost.append(all_SA_cost)
  # all_samples_DR_cost.append(all_DR_cost)


# save the results
# with open(f'stabilization_results_{n_trials}_M50.pkl', 'wb') as f:
#   pickle.dump({
#     'all_samples_Ks': all_samples_Ks,
#     'all_samples_SA_cost': all_samples_SA_cost,
#     'all_samples_DR_cost': all_samples_DR_cost
#   }, f)