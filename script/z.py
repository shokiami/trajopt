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
N = 15
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

def dfdx(x, u):
  return np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])

def dfdu(x, u):
  return 1.0 / MASS * np.vstack([np.zeros((3, 3)), np.eye(3)])

# OBSTACLE CONSTRAINTS
def g(x, u):
  P = np.hstack([np.eye(3), np.zeros((3, 3))])
  return np.array([r * r - np.dot(P @ x - c, P @ x - c) for (c, r) in OBS])

def dgdx(x, u):
  P = np.hstack([np.eye(3), np.zeros((3, 3))])
  return np.array([(-2.0 * (P @ x - c) @ P) for (c, r) in OBS])

def dgdu(x, u):
  return np.zeros((M_OBS, N_U))

# ALGORITHM
def integrate(x_i, u_i, u_f, T_i, T_f, F, dFdx, dFdu):
  n_z = N_X + M_OBS

  def concat(x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1):
    return np.hstack([x_prop, Ak.flatten(), Bk_0.flatten(), Bk_1.flatten(), Sk_0, Sk_1])

  def split(res):
    res_split = np.split(res, np.cumsum([n_z, n_z * n_z, n_z * N_U, n_z * N_U, n_z, n_z])[:-1])
    x_prop = res_split[0]
    Ak = res_split[1].reshape((n_z, n_z))
    Bk_0 = res_split[2].reshape((n_z, N_U))
    Bk_1 = res_split[3].reshape((n_z, N_U))
    Sk_0 = res_split[4]
    Sk_1 = res_split[5]

    return x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1

  def res_dot(tau, res):
    x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1 = split(res)

    u_foh = (1.0 - tau) * u_i + tau * u_f
    T_foh = (1.0 - tau) * T_i + tau * T_f
    f_xu = F(x_prop, u_foh)
    A_tau = T_foh * dFdx(x_prop, u_foh)
    B_tau = T_foh * dFdu(x_prop, u_foh)

    x_dot = T_foh * f_xu
    Ak_dot = A_tau @ Ak
    Bk_0_dot = A_tau @ Bk_0 + B_tau * (1.0 - tau)
    Bk_1_dot = A_tau @ Bk_1 + B_tau * tau
    Sk_0_dot = A_tau @ Sk_0 + f_xu * (1.0 - tau)
    Sk_1_dot = A_tau @ Sk_1 + f_xu * tau

    return concat(x_dot, Ak_dot, Bk_0_dot, Bk_1_dot, Sk_0_dot, Sk_1_dot)

  Ak_i = np.eye(n_z)
  Bk_0_i = np.zeros((n_z, N_U))
  Bk_1_i = np.zeros((n_z, N_U))
  Sk_0_i = np.zeros(n_z)
  Sk_1_i = np.zeros(n_z)
  res_i = concat(x_i, Ak_i, Bk_0_i, Bk_1_i, Sk_0_i, Sk_1_i)
  res = solve_ivp(res_dot, (0.0, 1.0), res_i).y[:, -1]
  x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1 = split(res)

  return x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1

def solve(x_ref, u_ref, T_ref, x_i, x_f, u_i, u_f, F, dFdx, dFdu):
  n_z = N_X + M_OBS
  x = cp.Variable((N + 1, n_z))
  u = cp.Variable((N + 1, N_U))
  T = cp.Variable(N + 1)
  gamma = cp.Variable(N + 1)
  eta = cp.Variable((N, n_z))

  cost = 0.0
  constr = []

  cost += U_W * cp.norm2(gamma)
  cost += T_W * cp.norm2(T)
  for i in range(N):
    cost += VIRTUAL_BUF * cp.norm1(eta[i])
  for i in range(N + 1):
    cost += 1.0 / (2.0 * X_STEP) * cp.norm2(x[i] - x_ref[i])
    cost += 1.0 / (2.0 * U_STEP) * cp.norm2(u[i] - u_ref[i])

  for i in range(N):
    # linearize and discretize dynamics and obstacle constraints
    x_prop, Ak, Bk_0, Bk_1, Sk_0, Sk_1 = integrate(x_ref[i], u_ref[i], u_ref[i + 1], T_ref[i], T_ref[i + 1], F, dFdx, dFdu)
    
    # dynamics constraints
    constr += [x[i + 1] + eta[i] == x_prop + Ak @ (x[i] - x_ref[i]) + Bk_0 @ (u[i] - u_ref[i]) + Bk_1 @ (u[i + 1] - u_ref[i + 1]) + Sk_0 * (T[i] - T_ref[i]) + Sk_1 * (T[i + 1] - T_ref[i + 1])]

    # obstacle constraints
    constr += [x[i + 1][N_X:] - x[i][N_X:] <= 1e-5]

  # control constraints
  for i in range(N + 1):
    constr += [cp.norm2(u[i]) <= gamma[i]]
    constr += [U_MIN <= gamma[i], gamma[i] <= U_MAX]
    constr += [np.cos(THETA_MAX) * gamma[i] <= u[i, 2]]

  # initial conditions
  constr += [x[0] == x_i, u[0] == u_i]

  # final conditions
  constr += [x[N] == x_f, u[N] == u_f]

  # time interval constraints
  for i in range(N + 1):
    constr += [T_MIN / N <= T[i], T[i] <= T_MAX / N]
    constr += [cp.norm2(T[i] - T_ref[i]) <= T_STEP]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.ECOS)

  return result, x.value, u.value, T.value

def total_time(T):
  return T[0] / 2 + np.sum(T[1:N]) + T[N] / 2

def ctcs():
  beta_i = np.zeros(M_OBS)
  beta_f = np.zeros(M_OBS)
  z_i = np.hstack([X_I, beta_i])
  z_f = np.hstack([X_F, beta_f])
  u_i = U_I
  u_f = U_F

  z_ref = np.linspace(z_i, z_f, N + 1)
  u_ref = np.zeros((N + 1, 3))
  T_ref = np.full(N + 1, T_MAX / N)

  def Z(z, u):
    x = z[:N_X]
    x_dot = f(x, u)
    B_dot = np.maximum(g(x, u), 0.0)**2
    return np.hstack([x_dot, B_dot])
  
  def dZdz(z, u):
    x = z[:N_X]
    A_tau = dfdx(x, u)
    g_xu = g(x, u)
    G_tau = np.zeros((M_OBS, N_X))
    G_tau = np.maximum(2.0 * g_xu.reshape(-1, 1) * dgdx(x, u), 0.0)
    return np.hstack([np.vstack([A_tau, G_tau]), np.zeros((N_X + M_OBS, M_OBS))])
  
  def dZdu(z, u):
    x = z[:N_X]
    B_tau = dfdu(x, u)
    g_xu = g(x, u)
    H_tau = np.zeros((M_OBS, N_U))
    H_tau = np.maximum(2.0 * g_xu.reshape(-1, 1) * dgdu(x, u), 0.0)
    return np.vstack([B_tau, H_tau])

  prev_cost = np.inf
  for i in range(4):
    cost, z, u, T = solve(z_ref, u_ref, T_ref, z_i, z_f, u_i, u_f, Z, dZdz, dZdu)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    z_ref = z
    u_ref = u
    T_ref = T
    print(total_time(T))

  return z[:, :3], u

if __name__ == '__main__':
  r, u = ctcs()
  plot(r, u, OBS)
