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
  r_dot = x[3:]
  v_dot = u / MASS - np.array([0.0, 0.0, 9.81])
  x_dot = np.hstack([r_dot, v_dot])
  return x_dot

def init_jacobians(t_i, t_f):
  def Ak_dot(t, Ak_flat):
    return (A(t) @ Ak_flat.reshape((6, 6))).flatten()

  def Bk_1_dot(t, Bk_1_flat):
    return (A(t) @ Bk_1_flat.reshape((6, 3)) + B(t) * (t_f - t) / (t_f - t_i)).flatten()

  def Bk_2_dot(t, Bk_2_flat):
    return (A(t) @ Bk_2_flat.reshape((6, 3)) + B(t) * (t - t_i) / (t_f - t_i)).flatten()

  Ak_i = np.eye(6)
  Bk_1_i = np.zeros((6, 3))
  Bk_2_i = np.zeros((6, 3))

  Ak = solve_ivp(Ak_dot, (t_i, t_f), Ak_i.flatten()).y[:, -1].reshape((6, 6))
  Bk_1 = solve_ivp(Bk_1_dot, (t_i, t_f), Bk_1_i.flatten()).y[:, -1].reshape((6, 3))
  Bk_2 = solve_ivp(Bk_2_dot, (t_i, t_f), Bk_2_i.flatten()).y[:, -1].reshape((6, 3))
  return Ak, Bk_1, Bk_2

def prop_dynamics(x_i, u_i, u_f, t_i, t_f):
  def x_dot(t, x):
    u_foh = (t_f - t) / (t_f - t_i) * u_i + (t - t_i) / (t_f - t_i) * u_f
    return f(x, u_foh)

  x_prop = solve_ivp(x_dot, (t_i, t_f), x_i).y[:,-1]
  return x_prop

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
  Ak, Bk_1, Bk_2 = init_jacobians(0.0, delta_t)
  for i in range(N):
    x_prop = prop_dynamics(x_ref[i], u_ref[i], u_ref[i + 1], 0.0, delta_t)
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
