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
action0 = 0.00
dt = 0.01

mass = 0.05
length = 0.2

true_mass = 0.024
true_length = 0.129

#0.024, 0.129

dx = 4
du = 1
Q = jnp.eye(dx)
R = 5*jnp.eye(du)
Sigma_w = jnp.eye(dx)

'''
  Define the cost function and its gradient
'''
@jit
def spec_rad(A):
  return jnp.max(jnp.abs(jnp.linalg.eigvals(A)))

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

def cost(K, A, B):
  ABK = A + jnp.einsum('...ij, jk -> ...ik', B, K)
  spec_rads = spec_rad(ABK)

  def compute_cost_fn(A, B):
    def true_fn(_):
      P = dlyap_direct((A+B@K).T, Q + K.T@R@K)
      return jnp.trace(P)
    
    def false_fn(_):
      return jnp.array(100000, dtype=jnp.float32)
    
    return jax.lax.cond(
      spec_rad(ABK) < 1,
      true_fn,
      false_fn,
      None
    )
    
  return compute_cost_fn(A, B)

def grad(K, A, B):
    P = dlyap_direct((A+B@K).T, Q + K.T@R@K)
    S = dlyap_direct(A+B@K, jnp.eye(dx))
    E = (R + B.T@P@B)@K + B.T@P@A
    return 2*E@S

def grad_descent(K, A, B, alpha, n_iterations):
    def body_fn(i, K):
        grad_k = grad(K, A, B)  # Use grad directly instead of avg_grad
        norm = jnp.linalg.norm(grad_k)
        clip_val = 1e3
        grad_k = jnp.where(norm > clip_val, grad_k * (clip_val/norm), grad_k)
        K = K - alpha*grad_k
        return K
    
    return jax.lax.fori_loop(0, n_iterations, body_fn, K)

# progressive discounting: find gamma satisfying 2.5*C_old < C_new < 8*C_old
def update_gamma(gamma, A, B, K):
  C_old = cost(K, jnp.sqrt(gamma)*A, jnp.sqrt(gamma)*B)

  gamma_lb = gamma
  gamma_ub = 1.0
  # bisection
  while gamma_ub - gamma_lb > 1e-4:
    gamma_mid = (gamma_ub + gamma_lb)/2
    C_new = cost(K, jnp.sqrt(gamma_mid)*A, jnp.sqrt(gamma_mid)*B)

    if C_new < 2.5*C_old:
      gamma_lb = gamma_mid
    elif C_new > 4*C_old:
      gamma_ub = gamma_mid
    else:
      gamma_lb = gamma_mid
      break
  return gamma_lb

alpha = 1e-3
n_iterations = 20
alpha = 1e-3
n_iterations = 20
n_itertions_final = 5

n_trials = 1


all_Ks = []
all_SA_cost = []
all_DR_cost = []

A, B = discretize_linearize_dynamics(state0, action0, dt, mass, length)
A_true, B_true = discretize_linearize_dynamics(state0, action0, dt, true_mass, true_length)

P = la.solve_discrete_are(A, B, Q, R)
K = -la.inv(B.T@P@B + R)@(B.T@P@A)

print('K: ', K)
print('spec_rad: ', spec_rad(A+B@K))
print('cost: ', cost(K, A, B))
print('cost_true: ', cost(K, A_true, B_true))
# for trial in range(n_trials):
#   # generate new subkey for each trial
#   print('A.shape: ', A.shape)
#   print('B.shape: ', B.shape)

#   rho = jnp.max(spec_rad(A))
#   gamma = min(0.9*rho**(-2), 1)

#   K = jnp.array([[0.0, 0.0, 0.0, 0.0]])

#   Ks = []
#   SA_cost = []
#   # DR_cost = [

#   # check if the spectral radius is less than 1 and gamma is less than 1
#   while gamma < 0.999:
#     K = grad_descent(K, jnp.sqrt(gamma)*A, jnp.sqrt(gamma)*B, alpha, n_iterations)
#     Ks.append(K)
#     SA_cost.append(cost(K, A, B))
#     # DR_cost.append(cost(K, eval_As, eval_Bs))
#     print('K: ', K)
#     print('spectral radius', spec_rad(A+B@K))
#   #   print('A + B@K', [A+B@K for (A,B) in zip(As,Bs)])
#     print('discounted spectral radius', spec_rad((jnp.sqrt(gamma)*A)+(jnp.sqrt(gamma)*B)@K))
#     print('iteration: ', len(Ks))
#     gamma = update_gamma(gamma, A, B, K)
#     print('gamma: ', gamma)
#   #   print('SA_cost: ', cost(K, As, Bs))
#   #   print('DR_cost: ', cost(K, eval_As, eval_Bs))
#   for i in range(50):
#     K = grad_descent(K, A, B, alpha, n_iterations)
#     Ks.append(K)
#     SA_cost.append(cost(K, A, B))
#     # DR_cost.append(cost(K, eval_As, eval_Bs))
#     print('iteration: ', len(Ks))
#     print('K: ', K)
#     print('SA_cost: ', cost(K, A, B))
#   #   print('DR_cost: ', cost(K, eval_As, eval_Bs))

#   print('final controller: ', K)
#   all_Ks.append(Ks)
#   all_SA_cost.append(SA_cost)
  # all_DR_cost.append(DR_cost)

# plt.figure(figsize=(10, 5))
# iterations = jnp.arange(len(all_SA_cost[0]))  
# plt.plot(iterations, all_SA_cost[0])
# plt.show()

  # all_samples_Ks.append(all_Ks)
  # all_samples_SA_cost.append(all_SA_cost)
  # all_samples_DR_cost.append(all_DR_cost)


# save the results
# with open(f'stabilization_results_{n_trials}_M50.pkl', 'wb') as f:
#   pickle.dump({
#     'all_samples_Ks': all_samples_Ks,
#     'all_samples_SA_cost': all_samples_SA_cost,
#     'all_samples_DR_cost': all_samples_DR_cost
#   }, f)