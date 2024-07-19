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
N = 30
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

N_X = 6
N_U = 3
M_OBS = len(OBS)

# DYNAMICS MODEL
def f(x, u):
  return np.hstack([x[3:], u / MASS - np.array([0.0, 0.0, 9.81])])

def dfdx(t):
  return np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])

def dfdu(t):
  return 1.0 / MASS * np.vstack([np.zeros((3, 3)), np.eye(3)])

# OBSTACLE CONSTRAINTS
def g(x, u):
  return np.array([r * r - np.dot(x[0:3] - c, x[0:3] - c) for (c, r) in OBS])

def dgdx(x, u):
  A = np.hstack([np.eye(3), np.zeros((3, 3))])
  return np.array([(-2.0 * (x[0:3] - c) @ A).T for (c, r) in OBS])

def dgdu(x, u):
  return np.zeros(M_OBS).T

# obstacle conditions
# for i in range(N + 1):
#   for j in range(M_OBS):
#     obs_c, obs_r = OBS[j]
#     constr += [obs_r * obs_r - cp.sum_squares(x_ref[i][0:3] - obs_c) + 2.0 * (x_ref[i][0:3] - obs_c).T @ (x_ref[i][0:3] - x[i][0:3]) <= eta[i, j]]
#     constr += [eta[i, j] >= 0.0]

# ALGORITHM
def integrate(x_i, u_i, u_f, T_i, T_f):
  def concat(x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1, beta_prop, Gk, Hk_0, Hk_1):
    return np.hstack([x_prop, Ak.flatten(), Bk_0.flatten(), Bk_1.flatten(), Sk_0, Sk_1, beta_prop, Gk.flatten(), Hk_0.flatten(), Hk_1.flatten()])

  def split(res):
    res_split = np.split(res, np.cumsum([N_X, N_X * N_X, N_X * N_U, N_X * N_U, N_X, N_X, M_OBS, M_OBS * N_X, M_OBS * N_U, M_OBS * N_U])[:-1])
    x_prop = res_split[0]
    Ak = res_split[1].reshape((N_X, N_X))
    Bk_0 = res_split[2].reshape((N_X, N_U))
    Bk_1 = res_split[3].reshape((N_X, N_U))
    Sk_0 = res_split[4]
    Sk_1 = res_split[5]
    beta_prop = res_split[6]
    Gk = res_split[7].reshape((M_OBS, N_X))
    Hk_0 = res_split[8].reshape((M_OBS, N_U))
    Hk_1 = res_split[9].reshape((M_OBS, N_U))

    return x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1, beta_prop, Gk, Hk_0, Hk_1

  def res_dot(tau, res):
    x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1, beta_prop, Gk, Hk_0, Hk_1 = split(res)

    u_foh = (1.0 - tau) * u_i + tau * u_f
    T_foh = (1.0 - tau) * T_i + tau * T_f
    f_xu = f(x_prop, u_foh)
    t = tau * T_foh
    A_tau = T_foh * dfdx(t)
    B_tau = T_foh * dfdu(t)

    g_xu = g(x_prop, u_foh)
    G_tau = np.zeros((M_OBS, N_X))
    H_tau = np.zeros((M_OBS, N_U))
    G_tau = np.maximum(2.0 * g_xu.reshape(-1, 1) * dgdx(x_prop, u_foh), 0.0)
    H_tau = np.maximum(2.0 * g_xu.reshape(-1, 1) * dgdu(x_prop, u_foh), 0.0)
  
    x_dot = T_foh * f_xu
    Ak_dot = A_tau @ Ak
    Bk_0_dot = A_tau @ Bk_0 + B_tau * (1.0 - tau)
    Bk_1_dot = A_tau @ Bk_1 + B_tau * tau
    Sk_0_dot = A_tau @ Sk_0 + f_xu * (1.0 - tau)
    Sk_1_dot = A_tau @ Sk_1 + f_xu * tau

    beta_dot = np.maximum(g_xu, 0.0)**2
    Gk_dot = G_tau @ Ak
    Hk_0_dot = G_tau @ Bk_0 + H_tau * (1.0 - tau)
    Hk_1_dot = G_tau @ Bk_1 + H_tau * tau

    return concat(x_dot, Ak_dot, Bk_0_dot, Bk_1_dot, Sk_0_dot, Sk_1_dot, beta_dot, Gk_dot, Hk_0_dot, Hk_1_dot)

  Ak_i = np.eye(N_X)
  Bk_0_i = np.zeros((N_X, N_U))
  Bk_1_i = np.zeros((N_X, N_U))
  Sk_0_i = np.zeros(N_X)
  Sk_1_i = np.zeros(N_X)
  Gk_i = np.zeros((M_OBS, N_X))
  Hk_0_i = np.zeros((M_OBS, N_U))
  Hk_1_i = np.zeros((M_OBS, N_U))
  res_i = concat(x_i, Ak_i, Bk_0_i, Bk_1_i, Sk_0_i, Sk_1_i, np.zeros(M_OBS), Gk_i, Hk_0_i, Hk_1_i)
  res = solve_ivp(res_dot, (0.0, 1.0), res_i).y[:, -1]
  x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1, beta, Gk, Hk_0, Hk_1 = split(res)

  return x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1, beta, Gk, Hk_0, Hk_1

def solve(x_ref, u_ref, T_ref):
  x = cp.Variable((N + 1, N_X))
  u = cp.Variable((N + 1, N_U))
  T = cp.Variable(N + 1)
  gamma = cp.Variable(N + 1)
  eta = cp.Variable((N, M_OBS))

  cost = 0.0
  constr = []

  cost += U_W * cp.norm2(gamma)
  cost += T_W * cp.norm2(T)
  for i in range(N):
    cost += VIRTUAL_BUF * cp.norm2(eta[i])
  for i in range(N + 1):
    cost += 1.0 / (2.0 * X_STEP) * cp.norm2(x[i] - x_ref[i])
    cost += 1.0 / (2.0 * U_STEP) * cp.norm2(u[i] - u_ref[i])

  for i in range(N):
    # linearize and discretize dynamics and obstacle constraints
    x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1, beta, Gk, Hk_0, Hk_1 = integrate(x_ref[i], u_ref[i], u_ref[i + 1], T_ref[i], T_ref[i + 1])
    
    # dynamics constraints
    constr += [x[i + 1] == x_prop + Ak @ (x[i] - x_ref[i]) + Bk_0 @ (u[i] - u_ref[i]) + Bk_1 @ (u[i + 1] - u_ref[i + 1]) + Sk_0 * (T[i] - T_ref[i]) + Sk_1 * (T[i + 1] - T_ref[i + 1])]

    # obstacle constraints
    constr += [beta + Gk @ (x[i] - x_ref[i]) + Hk_0 @ (u[i] - u_ref[i]) + Hk_1 @ (u[i + 1] - u_ref[i + 1]) <= eta[i]]
    constr += [eta[i] >= 0.0]

  # control constraints
  for i in range(N + 1):
    constr += [cp.norm2(u[i]) <= gamma[i]]
    constr += [U_MIN <= gamma[i], gamma[i] <= U_MAX]
    constr += [np.cos(THETA_MAX) * gamma[i] <= u[i, 2]]

  # initial conditions
  constr += [x[0] == X_I, u[0] == U_I]

  # final conditions
  constr += [x[N] == X_F, u[N] == U_F]

  # time interval constraints
  for i in range(N + 1):
    constr += [T_MIN / N <= T[i], T[i] <= T_MAX / N]
    constr += [cp.norm2(T[i] - T_ref[i]) <= T_STEP]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.ECOS, max_iters=1000)

  return result, x.value, u.value, T.value

def total_time(T):
  return T[0] / 2 + np.sum(T[1:N]) + T[N] / 2

if __name__ == '__main__':
  x_ref = np.linspace(X_I, X_F, N + 1)
  u_ref = np.zeros((N + 1, 3))
  T_ref = np.full(N + 1, T_MAX / N)

  prev_cost = np.inf
  for i in range(4):
    cost, x, u, T = solve(x_ref, u_ref, T_ref)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    x_ref = x
    u_ref = u
    T_ref = T
    print(total_time(T))

  pos = x[:, :3]
  control = u
  plot(pos, control, OBS)
