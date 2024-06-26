from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

VIRTUAL_BUF = 1e4
STEP_SIZE = 100.0
CONV_EPS = 1e-1

X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
U_I = [0.0, 0.0, 9.81]
U_F = [0.0, 0.0, 9.81]
N = 25
T_F = 3.0
U_MIN = 1.0
U_MAX = 20.0
THETA_MAX = np.pi / 4.0
MASS = 1.0

OBS = [
  ((2.5, 1.5, 2.5), 2.0),
  ((5.0, 6.0, 4.0), 3.0),
  ((10.0, 8.0, 7.5), 2.5),
  ((1.0, 9.0, 9.0), 3.5),
  ((9.0, 1.0, 3.0), 1.5),
]

def A(t):
  return np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])

def B(t):
  return 1.0 / MASS * np.vstack([np.zeros((3, 3)), np.eye(3)])

def f(x, u):
  return np.hstack([x[3:], u / MASS - np.array([0.0, 0.0, 9.81])])

def integrate(x_i, u_i, u_f, t_i, t_f):
  def concat(x_prop, Ak, Bk_1, Bk_2):
    return np.hstack([x_prop, Ak.flatten(), Bk_1.flatten(), Bk_2.flatten()])

  def split(res):
    x_prop = res[0:6]
    Ak = res[6:42].reshape((6, 6))
    Bk_1 = res[42:60].reshape((6, 3))
    Bk_2 = res[60:78].reshape((6, 3))

    return x_prop, Ak, Bk_1, Bk_2

  def res_dot(t, res):
    x_prop, Ak, Bk_1, Bk_2 = split(res)

    x_dot = f(x_prop, (t_f - t) / (t_f - t_i) * u_i + (t - t_i) / (t_f - t_i) * u_f)
    Ak_dot = A(t) @ Ak
    Bk_1_dot = A(t) @ Bk_1 + B(t) * (t_f - t) / (t_f - t_i)
    Bk_2_dot = A(t) @ Bk_2 + B(t) * (t - t_i) / (t_f - t_i)

    return concat(x_dot, Ak_dot, Bk_1_dot, Bk_2_dot)

  Ak_i = np.eye(6)
  Bk_1_i = np.zeros((6, 3))
  Bk_2_i = np.zeros((6, 3))
  res_i = concat(x_i, Ak_i, Bk_1_i, Bk_2_i)
  res = solve_ivp(res_dot, (t_i, t_f), res_i).y[:, -1]
  x_prop, Ak, Bk_1, Bk_2 = split(res)

  return x_prop, Ak, Bk_1, Bk_2

def solve(x_ref, u_ref):
  x = cp.Variable((N + 1, 6))
  u = cp.Variable((N + 1, 3))
  gamma = cp.Variable(N + 1)
  eta = cp.Variable((N + 1, len(OBS)))

  cost = 0.0
  constr = []

  cost += cp.sum_squares(gamma)
  for i in range(N + 1):
    cost += VIRTUAL_BUF * cp.norm1(eta[i])
    cost += 1.0 / (2.0 * STEP_SIZE) * cp.norm2(x[i] - x_ref[i])

  # dynamics constraints
  delta_t = T_F / N
  for i in range(N):
    x_prop, Ak, Bk_1, Bk_2 = integrate(x_ref[i], u_ref[i], u_ref[i + 1], 0.0, delta_t)
    constr += [x[i + 1] == x_prop + Ak @ (x[i] - x_ref[i]) + Bk_1 @ (u[i] - u_ref[i]) + Bk_2 @ (u[i + 1] - u_ref[i + 1])]

  # control constraints
  for i in range(N + 1):
    constr += [cp.norm(u[i]) <= gamma[i]]
    constr += [U_MIN <= gamma[i], gamma[i] <= U_MAX]
    constr += [np.cos(THETA_MAX) * gamma[i] <= u[i, 2]]

  # initial conditions
  constr += [x[0] == X_I, u[0] == U_I]

  # final conditions
  constr += [x[N] == X_F, u[N] == U_F]

  # obstacle conditions
  for i in range(N + 1):
    for j in range(len(OBS)):
      obs_c, obs_r = OBS[j]
      d = x_ref[i][:3] - obs_c
      constr += [obs_r * obs_r - d.T @ d + 2.0 * d.T @ (x_ref[i][:3] - x[i][:3]) <= eta[i, j]]
      constr += [eta[i, j] >= 0.0]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.ECOS)

  return result, x.value, u.value

if __name__ == '__main__':
  x_ref = np.linspace(X_I, X_F, N + 1)
  u_ref = np.zeros((N + 1, 3))

  prev_cost = np.inf
  while True:
    cost, x, u = solve(x_ref, u_ref)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    x_ref = x
    u_ref = u
    print(cost)

  pos = x[:,:3]
  control = u
  plot(pos, control, OBS)
