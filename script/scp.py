from viz import plot
import cvxpy as cp
import numpy as np

U_W = 1e2
VIRTUAL_BUF = 1e4
STEP_SIZE = 1e2
CONV_EPS = 1e-2

R_I = [0.0, 0.0, 0.0]
R_F = [10.0, 10.0, 10.0]
V_I = [0.0, 0.0, 0.0]
V_F = [0.0, 0.0, 0.0]
A_I = [0.0, 0.0, 0.0]
A_F = [0.0, 0.0, 0.0]
N = 20
T_F = 4.0
U_MIN = 1.0
U_MAX = 20.0
THETA_MAX = np.pi / 4.0
MASS = 1.0

OBS = [
  ((2.5, 1.5, 2.5), 2.0),
  ((5.0, 5.5, 5.0), 3.0),
  ((7.5, 8.0, 9.5), 2.5),
  ((1.0, 9.0, 9.0), 3.5),
  ((9.0, 1.0, 3.0), 1.5),
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

  cost += U_W * cp.norm2(gamma)
  for i in range(N + 1):
    cost += VIRTUAL_BUF * cp.norm2(eta[i])
    cost += 1.0 / (2.0 * STEP_SIZE) * cp.norm2(r[i] - r_ref[i])

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

  # obstacle conditions
  for i in range(N + 1):
    for j in range(len(OBS)):
      obs_c, obs_r = OBS[j]
      constr += [obs_r * obs_r - cp.sum_squares(r_ref[i] - obs_c) + 2.0 * (r_ref[i] - obs_c).T @ (r_ref[i] - r[i]) <= eta[i, j]]
      constr += [eta[i, j] >= 0.0]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.ECOS)
  return result, r.value, u.value

if __name__ == '__main__':
  r_ref = np.linspace(R_I, R_F, N + 1)

  prev_cost = np.inf
  while True:
    cost, r, u = solve(r_ref)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    r_ref = r
    print(cost)

  plot(r, u, OBS)
