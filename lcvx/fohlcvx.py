from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

N = 8
X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
T_F = 4.0
RHO_MIN = 4.0
RHO_MAX = 6.0
RHO_FEAS = 5.0
CONV_EPS = 1e-1
VIOL_EPS = 1e-4

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

def solve(rho, delta):
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
    constr += [cp.norm2(u[i]) <= RHO_MAX]
    constr += [rho <= sigma[i]]
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
  rho_low = RHO_MIN
  rho_high = RHO_FEAS
  while rho_high - rho_low > CONV_EPS:
    rho = (rho_low + rho_high) / 2
    delta = 2 * np.sqrt(rho**2 - RHO_MIN**2)
    cost, x, u = solve(rho, delta)
    if cost != np.inf and np.all(np.linalg.norm(u, axis=1) > rho - VIOL_EPS):
      rho_high = rho
    else:
      rho_low = rho
  delta = 2 * np.sqrt(rho_high**2 - RHO_MIN**2)
  cost, x, u = solve(rho_high, delta)
  return x, u

if __name__ == '__main__':
  x, u = fohlcvx()
  plot(x, u, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, True)

  # delta = 4
  # rho_min = np.sqrt(0.25 * delta**2 + RHO_MIN**2)
  # cost, x, u = solve(rho_min, RHO_MAX, delta)
  # plot(x, u, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, True)
  # exit()

  # for rho_min in np.arange(4.0, 6.0, 0.1):
  #   delta = 2 * np.sqrt(rho_min**2 - RHO_MIN**2)
  #   cost, x, u = solve(rho_min, RHO_MAX, delta)
  #   print(cost != np.inf, np.all(np.linalg.norm(u, axis=1) > rho_min - VIOL_EPS) if cost != np.inf else False)
  # exit()

  # for delta in np.arange(0.0, 7.2, 0.1):
  #   rho_min = np.sqrt(0.25 * delta**2 + RHO_MIN**2)
  #   cost, x, u = solve(rho_min, RHO_MAX, delta)
  #   print(cost != np.inf, np.all(np.linalg.norm(u, axis=1) > rho_min - VIOL_EPS) if cost != np.inf else False)
  # exit()

  # rhos = np.arange(4.0, 5.0, 0.1)
  # delta_mins = []
  # delta_opts = []
  # delta_maxes = []
  # for rho in rhos:
  #   print(f'rho: {rho}')
  #   delta_min = 0
  #   delta_opt = 0
  #   cost, x, u = solve(rho, RHO_MAX, np.inf)
  #   delta_max = np.max(np.linalg.norm(np.diff(u), axis=1))
  #   for delta in np.arange(0.0, 20.0, 0.1):
  #     cost, x, u = solve(rho, RHO_MAX, delta)
  #     if cost != np.inf and delta_min == 0:
  #       delta_min = delta
  #       print(f'delta_min: {delta_min}')
  #     if cost != np.inf:
  #       if np.all(np.linalg.norm(u, axis=1) > rho - VIOL_EPS) and delta_opt == 0:
  #         delta_opt = delta
  #         print(f'delta_opt: {delta_opt}')
  #         break
  #   delta_mins.append(delta_min)
  #   delta_opts.append(delta_opt)
  #   delta_maxes.append(delta_max)
  # plt.plot(rhos, delta_mins)
  # plt.plot(rhos, delta_opts)
  # plt.plot(rhos, delta_maxes)
  # plt.plot(rhos, 2 * np.sqrt(rhos**2 - RHO_MIN**2))
  # plt.show()
  # exit()
