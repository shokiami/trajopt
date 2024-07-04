from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

U_W = 1e-1
T_W = 1e3
VIRTUAL_BUF = 1e4
X_STEP = 1e2
U_STEP = 1e2
T_STEP = 1e-1
CONV_EPS = 1e-2

X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
U_I = [0.0, 0.0, 9.81]
U_F = [0.0, 0.0, 9.81]
N = 20
T_MIN = 1.0
T_MAX = 4.0
U_MIN = 1.0
U_MAX = 20.0
THETA_MAX = np.pi / 4.0
MASS = 1.0

OBS = [
  ((3.0, 2.0, 2.3), 2.0),
  ((4.8, 7.0, 6.0), 3.0),
  ((9.0, 8.0, 10.0), 1.0),
]

# DYNAMICS MODEL
def f(x, u):
  return np.hstack([x[3:6], u / MASS - np.array([0.0, 0.0, 9.81])])

def A(t):
  return np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])

def B(t):
  return 1.0 / MASS * np.vstack([np.zeros((3, 3)), np.eye(3)])

# ALGORITHM
def integrate(x_i, u_i, u_f, Tk):
  def concat(x_prop, Ak, Bk_1, Bk_2, Sk):
    return np.hstack([x_prop, Ak.flatten(), Bk_1.flatten(), Bk_2.flatten(), Sk])

  def split(res):
    x_prop = res[0:6]
    Ak = res[6:42].reshape((6, 6))
    Bk_1 = res[42:60].reshape((6, 3))
    Bk_2 = res[60:78].reshape((6, 3))
    Sk = res[78:84]

    return x_prop, Ak, Bk_1, Bk_2, Sk

  def res_dot(tau, res):
    x_prop, Ak, Bk_1, Bk_2, Sk = split(res)

    u_foh = (1.0 - tau) * u_i + tau * u_f
    f_xu = f(x_prop, u_foh)
    t = tau * Tk
    A_tau = Tk * A(t)
    B_tau = Tk * B(t)
  
    x_dot = Tk * f_xu
    Ak_dot = A_tau @ Ak
    Bk_1_dot = A_tau @ Bk_1 + B_tau * (1.0 - tau)
    Bk_2_dot = A_tau @ Bk_2 + B_tau * tau
    Sk_dot = A_tau @ Sk + f_xu

    return concat(x_dot, Ak_dot, Bk_1_dot, Bk_2_dot, Sk_dot)

  Ak_i = np.eye(6)
  Bk_1_i = np.zeros((6, 3))
  Bk_2_i = np.zeros((6, 3))
  Sk_i = np.zeros(6)
  res_i = concat(x_i, Ak_i, Bk_1_i, Bk_2_i, Sk_i)
  res = solve_ivp(res_dot, (0.0, 1.0), res_i).y[:, -1]
  x_prop, Ak, Bk_1, Bk_2, Sk = split(res)

  return x_prop, Ak, Bk_1, Bk_2, Sk

def solve(x_ref, u_ref, T_ref):
  x = cp.Variable((N + 1, 6))
  u = cp.Variable((N + 1, 3))
  T = cp.Variable(N)
  gamma = cp.Variable(N + 1)
  eta = cp.Variable((N + 1, len(OBS)))

  cost = 0.0
  constr = []

  cost += U_W * cp.norm2(gamma)
  cost += T_W * cp.norm2(T)
  for i in range(N + 1):
    cost += VIRTUAL_BUF * cp.norm2(eta[i])
    cost += 1.0 / (2.0 * X_STEP) * cp.norm2(x[i] - x_ref[i])
    cost += 1.0 / (2.0 * U_STEP) * cp.norm2(u[i] - u_ref[i])

  # dynamics constraints
  for i in range(N):
    x_prop, Ak, Bk_1, Bk_2, Sk = integrate(x_ref[i], u_ref[i], u_ref[i + 1], T_ref[i])
    constr += [x[i + 1] == x_prop + Ak @ (x[i] - x_ref[i]) + Bk_1 @ (u[i] - u_ref[i]) + Bk_2 @ (u[i + 1] - u_ref[i + 1]) + Sk * (T[i] - T_ref[i])]

  # control constraints
  for i in range(N + 1):
    constr += [cp.norm2(u[i]) <= gamma[i]]
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
      constr += [obs_r * obs_r - cp.sum_squares(x_ref[i][0:3] - obs_c) + 2.0 * (x_ref[i][0:3] - obs_c).T @ (x_ref[i][0:3] - x[i][0:3]) <= eta[i, j]]
      constr += [eta[i, j] >= 0.0]

  # time interval constraints
  for i in range(N):
    constr += [T_MIN / N <= T[i], T[i] <= T_MAX / N]
    constr += [cp.norm2(T[i] - T_ref[i]) <= T_STEP]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.ECOS)

  return result, x.value, u.value, T.value

if __name__ == '__main__':
  x_ref = np.linspace(X_I, X_F, N + 1)
  u_ref = np.zeros((N + 1, 3))
  T_ref = np.full(N, T_MAX / N)

  prev_cost = np.inf
  while True:
    cost, x, u, T = solve(x_ref, u_ref, T_ref)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    x_ref = x
    u_ref = u
    T_ref = T
    print(np.sum(T))

  pos = x[:, 0:3]
  control = u
  plot(pos, control, OBS)
