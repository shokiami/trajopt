from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

T_WEIGHT = 1e2
VIRTUAL_CONTROL = 1e4
X_TRUST = 1e2
U_TRUST = 1e2
T_STEP = 1e-1
CONV_EPS = 1e-2

X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
U_I = [0.0, 0.0, 9.81]
U_F = [0.0, 0.0, 9.81]
N = 5
T_MIN = 2.0
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
  return np.array([r * r - np.linalg.norm(P @ x - c)**2 for (c, r) in OBS])

def dgdx(x, u):
  P = np.hstack([np.eye(3), np.zeros((3, 3))])
  return np.array([(-2.0 * P.T @ (P @ x - c)) for (c, r) in OBS])

def dgdu(x, u):
  return np.zeros((len(OBS), N_U))

# ALGORITHM
def integrate(x_i, u_i, u_f, Tk, F, dFdx, dFdu):
  n_z = N_X + 1

  def concat(x_prop, Ak, Bk_0, Bk_1, Sk):
    return np.hstack([x_prop, Ak.flatten(), Bk_0.flatten(), Bk_1.flatten(), Sk])

  def split(res):
    res_split = np.split(res, np.cumsum([n_z, n_z * n_z, n_z * N_U, n_z * N_U, n_z])[:-1])
    x_prop = res_split[0]
    Ak = res_split[1].reshape((n_z, n_z))
    Bk_0 = res_split[2].reshape((n_z, N_U))
    Bk_1 = res_split[3].reshape((n_z, N_U))
    Sk = res_split[4]

    return x_prop, Ak, Bk_0, Bk_1, Sk

  def res_dot(tau, res):
    x_prop, Ak, Bk_0, Bk_1, Sk = split(res)

    u_foh = (1.0 - tau) * u_i + tau * u_f
    f_xu = F(x_prop, u_foh)
    A_tau = Tk * dFdx(x_prop, u_foh)
    B_tau = Tk * dFdu(x_prop, u_foh)

    x_dot = Tk * f_xu
    Ak_dot = A_tau @ Ak
    Bk_0_dot = A_tau @ Bk_0 + B_tau * (1.0 - tau)
    Bk_1_dot = A_tau @ Bk_1 + B_tau * tau
    Sk_dot = A_tau @ Sk + f_xu

    return concat(x_dot, Ak_dot, Bk_0_dot, Bk_1_dot, Sk_dot)

  Ak_i = np.eye(n_z)
  Bk_0_i = np.zeros((n_z, N_U))
  Bk_1_i = np.zeros((n_z, N_U))
  Sk_i = np.zeros(n_z)
  res_i = concat(x_i, Ak_i, Bk_0_i, Bk_1_i, Sk_i)
  res = solve_ivp(res_dot, (0.0, 1.0), res_i).y[:, -1]
  x_prop, Ak, Bk_0, Bk_1, Sk = split(res)

  return x_prop, Ak, Bk_0, Bk_1, Sk

def solve(x_ref, u_ref, T_ref, x_i, x_f, u_i, u_f, F, dFdx, dFdu):
  n_z = N_X + 1
  x = cp.Variable((N + 1, n_z))
  u = cp.Variable((N + 1, N_U))
  T = cp.Variable(N)
  gamma = cp.Variable(N + 1)
  eta = cp.Variable((N, n_z))

  cost = 0.0
  constr = []

  cost += T_WEIGHT * cp.norm2(T)
  cost += VIRTUAL_CONTROL * cp.norm1(eta)
  for i in range(N + 1):
    cost += X_TRUST * cp.norm2(x[i] - x_ref[i])
    cost += U_TRUST * cp.norm2(u[i] - u_ref[i])

  for i in range(N):
    # linearize and discretize dynamics and obstacle constraints
    x_prop, Ak, Bk_0, Bk_1, Sk = integrate(x_ref[i], u_ref[i], u_ref[i + 1], T_ref[i], F, dFdx, dFdu)

    # dynamics constraints
    constr += [x[i + 1] == x_prop + Ak @ (x[i] - x_ref[i]) + Bk_0 @ (u[i] - u_ref[i]) + Bk_1 @ (u[i + 1] - u_ref[i + 1]) + Sk * (T[i] - T_ref[i]) + eta[i]]

    # obstacle constraints
    constr += [x[i][N_X:] == 0.0]
    constr += [x[i + 1][N_X:] == 0.0]

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
  for i in range(N):
    constr += [T_MIN / N <= T[i], T[i] <= T_MAX / N]
    constr += [cp.norm2(T[i] - T_ref[i]) <= T_STEP]

  prob = cp.Problem(cp.Minimize(cost), constr)
  result = prob.solve(solver=cp.CLARABEL)

  return result, x.value, u.value, T.value

def single_shot(x_i, u, T):
  T_cumsum = np.cumsum(T)

  def x_dot(t, x):
    i = np.searchsorted(T_cumsum, t)
    tau = (t - (T_cumsum[i - 1] if i > 0 else 0.0)) / T[i]
    u_foh = (1.0 - tau) * u[i] + tau * u[i + 1]
    return f(x, u_foh)
  
  res = solve_ivp(x_dot, (0.0, T_cumsum[-1]), x_i, max_step=1e-2).y[:]
  r_prop = res[:3].swapaxes(0, 1)
  return r_prop

def ctcs():
  z_i = np.hstack([X_I, 0.0])
  z_f = np.hstack([X_F, 0.0])
  u_i = U_I
  u_f = U_F

  z_ref = np.linspace(z_i, z_f, N + 1)
  u_ref = np.zeros((N + 1, 3))
  T_ref = np.full(N, T_MAX / N)

  def Z(z, u):
    x = z[:N_X]
    B_dot = np.sum(np.maximum(g(x, u), 0.0)**2, axis=0)
    return np.hstack([f(x, u), B_dot])
  
  def dZdz(z, u):
    x = z[:N_X]
    g_xu = g(x, u)
    mask = g_xu > 0.0
    dBdx = 2.0 * np.sum(g_xu[mask].reshape(-1, 1) * dgdx(x, u)[mask], axis=0)
    return np.hstack([np.vstack([dfdx(x, u), dBdx]), np.zeros((N_X + 1, 1))])

  def dZdu(z, u):
    x = z[:N_X]
    g_xu = g(x, u)
    mask = g_xu > 0.0
    dBdu = 2.0 * np.sum(g_xu[mask].reshape(-1, 1) * dgdu(x, u)[mask], axis=0)
    return np.vstack([dfdu(x, u), dBdu])

  prev_cost = np.inf
  while True:
    cost, z, u, T = solve(z_ref, u_ref, T_ref, z_i, z_f, u_i, u_f, Z, dZdz, dZdu)
    if np.abs(cost - prev_cost) < CONV_EPS:
      break
    prev_cost = cost
    z_ref = z
    u_ref = u
    T_ref = T
    print(cost)

  r = z[:, :3]
  r_prop = single_shot(X_I, u, T)
  print(f'final time: {np.sum(T)}')

  return r, u, r_prop

if __name__ == '__main__':
  r, u, r_prop = ctcs()
  plot(r, u, OBS, r_prop)
