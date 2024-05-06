from viz import plot
import cvxpy as cp
import numpy as np

VIRTUAL_BUF = 1e4
CONV_EPS = 1e-2

r_i = np.array([0.0, 0.0, 0.0])
r_f = np.array([10.0, 10.0, 10.0])
v_i = np.array([0.0, 0.0, 0.0])
v_f = np.array([0.0, 0.0, 0.0])
a_i = np.array([0.0, 0.0, 0.0])
a_f = np.array([0.0, 0.0, 0.0])
n = 50
t_f = 14.0
u_min = 1.0
u_max = 10.0
theta_max = np.pi / 4
mass = 1.0

obs_c = np.array([
  [3.0, 2.0, 3.0],
  [5.0, 9.0, 9.0]
])
obs_r = np.array([2.0, 4.0])

def solve(r_ref):
  r = cp.Variable((n + 1, 3))
  v = cp.Variable((n + 1, 3))
  a = cp.Variable((n + 1, 3))
  u = cp.Variable((n + 1, 3))
  gamma = cp.Variable(n + 1)
  eta = cp.Variable((n + 1, len(obs_c)))

  cost = cp.sum_squares(gamma) + VIRTUAL_BUF * cp.sum_squares(eta)
  constr = []

  # dynamics
  g = np.array([0, 0, 9.81])
  for i in range(n + 1):
    constr += [a[i] == u[i] / mass - g]
  dt = t_f / n
  dt2 = dt * dt
  for i in range(n):
    constr += [r[i + 1] == r[i] + dt * v[i] + dt2 / 3.0 * a[i] + dt2 / 6.0 * a[i + 1]]
    constr += [v[i + 1] == v[i] + dt / 2.0 * a[i] + dt / 2.0 * a[i + 1]]

  # control constraints
  for i in range(n + 1):
    constr += [cp.norm(u[i]) <= gamma[i]]
    constr += [u_min <= gamma[i], gamma[i] <= u_max]
    constr += [np.cos(theta_max) * gamma[i] <= u[i, 2]]

  # initial conditions
  constr += [r[0] == r_i, v[0] == v_i, a[0] == a_i]

  # final conditions
  constr += [r[n] == r_f, v[n] == v_f, a[n] == a_f]

  # obstacle conditions
  for i in range(n + 1):
    for j in range(len(obs_c)):
      d = r_ref[i] - obs_c[j]
      constr += [obs_r[j] * obs_r[j] - d.T @ d + 2.0 * d.T @ (r_ref[i] - r[i]) <= eta[i, j]]
      constr += [eta[i, j] >= 0.0]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.ECOS)
  print(result)
  return result, r.value, u.value

if __name__ == '__main__':
  r_ref = np.linspace(r_i, r_f, n + 1)

  prev_cost = np.inf
  while True:
    cost, r, u = solve(r_ref)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost

  plot(r, u, obs_c, obs_r)
