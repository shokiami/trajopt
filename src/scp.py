from viz import plot
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

U_WEIGHT = 1e0
VIRTUAL_BUF = 1e8
R_TRUST = 1e-4
CONV_EPS = 1e-2

R_I = [0.0, 0.0, 0.0]
R_F = [10.0, 10.0, 10.0]
V_I = [0.0, 0.0, 0.0]
V_F = [0.0, 0.0, 0.0]
A_I = [0.0, 0.0, 0.0]
A_F = [0.0, 0.0, 0.0]
N = 10
T_F = 4.0
U_MIN = 1.0
U_MAX = 20.0
THETA_MAX = np.pi / 4.0
MASS = 1.0

OBS = [
  ((3.0, 2.0, 2.3), 2.0),
  ((4.8, 7.0, 5.5), 3.0),
  ((9.0, 8.0, 10.0), 1.0),
]

def solve(r_ref):
  r = cp.Variable((N + 1, 3))
  v = cp.Variable((N + 1, 3))
  a = cp.Variable((N + 1, 3))
  u = cp.Variable((N + 1, 3))
  gamma = cp.Variable(N + 1)
  eta = cp.Variable((N + 1, len(OBS)))

  cost = 0.0
  constr = []

  cost += U_WEIGHT * cp.norm2(gamma)
  cost += VIRTUAL_BUF * cp.norm1(eta)
  for i in range(N + 1):
    cost += R_TRUST * cp.norm2(r[i] - r_ref[i])

  # dynamics constraints
  g = [0.0, 0.0, 9.81]
  for i in range(N + 1):
    constr += [a[i] == u[i] / MASS - g]
  delta_t = T_F / N
  delta_t_2 = delta_t * delta_t
  for i in range(N):
    constr += [r[i + 1] == r[i] + delta_t * v[i] + delta_t_2 / 3.0 * a[i] + delta_t_2 / 6.0 * a[i + 1]]
    constr += [v[i + 1] == v[i] + delta_t / 2.0 * a[i] + delta_t / 2.0 * a[i + 1]]

  # control constraints
  for i in range(N + 1):
    constr += [cp.norm2(u[i]) <= gamma[i]]
    constr += [U_MIN <= gamma[i], gamma[i] <= U_MAX]
    constr += [np.cos(THETA_MAX) * gamma[i] <= u[i, 2]]

  # initial conditions
  constr += [r[0] == R_I, v[0] == V_I, a[0] == A_I]

  # final conditions
  constr += [r[N] == R_F, v[N] == V_F, a[N] == A_F]

  # obstacle constraints
  for i in range(N + 1):
    for j in range(len(OBS)):
      obs_c, obs_r = OBS[j]
      constr += [obs_r * obs_r - cp.sum_squares(r_ref[i] - obs_c) + 2.0 * (r_ref[i] - obs_c).T @ (r_ref[i] - r[i]) <= eta[i, j]]
      constr += [eta[i, j] >= 0.0]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.CLARABEL)
  return result, r.value, u.value

if __name__ == '__main__':
  r_ref = np.linspace(R_I, R_F, N + 1)

  ax = plt.figure('trajopt').add_subplot(projection='3d')
  ax.set_title('Optimal Trajectory')

  prev_cost = np.inf
  while True:
    cost, r, u = solve(r_ref)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    r_ref = r
    
    plot(ax, r, u, OBS)
    plt.pause(0.01)
    print(f'cost: {cost}')
  
  plt.show()
