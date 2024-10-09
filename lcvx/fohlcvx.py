from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

N = 10
X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
T_F = 4.0
RHO_MIN = 4.0
RHO_MAX = 6.0
CONV_EPS = 1e-1
VIOL_EPS = 1e-3

A = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])
B = np.vstack([np.zeros((3, 3)), np.eye(3)])

def f(x, u):
  return A @ x + B @ u

def integrate(dt):
  def concat(Ak, Bk_0, Bk_1):
    return np.hstack([Ak.flatten(), Bk_0.flatten(), Bk_1.flatten()])

  def split(res):
    Ak = res[0:36].reshape((6, 6))
    Bk_0 = res[36:54].reshape((6, 3))
    Bk_1 = res[54:72].reshape((6, 3))

    return Ak, Bk_0, Bk_1

  def res_dot(t, res):
    Ak, Bk_0, Bk_1 = split(res)

    Ak_dot = A @ Ak
    Bk_0_dot = A @ Bk_0 + B * (dt - t) / dt
    Bk_1_dot = A @ Bk_1 + B * t / dt

    return concat(Ak_dot, Bk_0_dot, Bk_1_dot)

  Ak_i = np.eye(6)
  Bk_0_i = np.zeros((6, 3))
  Bk_1_i = np.zeros((6, 3))
  res_i = concat(Ak_i, Bk_0_i, Bk_1_i)
  res = solve_ivp(res_dot, (0.0, dt), res_i).y[:, -1]
  Ak, Bk_0, Bk_1 = split(res)

  return Ak, Bk_0, Bk_1

def solve(rho_min, rho_max, delta):
  x = cp.Variable((N + 1, 6))
  u = cp.Variable((N + 1, 3))
  sigma = cp.Variable(N + 1)

  cost = 0.0
  constr = []

  cost += cp.norm2(sigma)

  # dynamics constraints
  Ak, Bk_0, Bk_1 = integrate(T_F / N)
  for i in range(N):
    constr += [x[i + 1] == Ak @ x[i] + Bk_0 @ u[i] + Bk_1 @ u[i + 1]]

  # control constraints
  for i in range(N + 1):
    constr += [cp.norm2(u[i]) <= sigma[i]]
    constr += [cp.norm2(u[i]) <= rho_max]
    constr += [rho_min <= sigma[i]]
  for i in range(N):
    constr += [cp.norm2(u[i + 1] - u[i]) <= delta]

  # initial conditions
  constr += [x[0] == X_I]

  # final conditions
  constr += [x[N] == X_F]

  prob = cp.Problem(cp.Minimize(cost), constr)
  cost = prob.solve(solver=cp.CLARABEL)

  return cost, x.value, u.value

def fohlcvx():
  cost, x, u = solve(RHO_MIN, RHO_MAX, np.inf)
  delta_hi = np.max(np.linalg.norm(np.diff(u), axis=1))
  delta_lo = 0.0
  while delta_hi - delta_lo > CONV_EPS:
    delta = (delta_hi + delta_lo) / 2.0
    rho_min = np.sqrt(0.25 * delta**2 + RHO_MIN**2)
    cost, x, u = solve(rho_min, RHO_MAX, delta)
    if cost != np.inf and np.all(np.linalg.norm(u, axis=1) > rho_min - VIOL_EPS):
      delta_hi = delta
    else:
      delta_lo = delta
  return x, u

if __name__ == '__main__':
  x, u = fohlcvx()
  plot(x, u, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, True)
