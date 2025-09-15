from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

N = 16
X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
T_F = 4.0
RHO_MIN = 4.0
RHO_MAX = 6.0
CONV_EPS = 1e-1
VIOL_EPS = 1e-3
N_X = 6
N_U = 3

A = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])
B = np.vstack([np.zeros((3, 3)), np.eye(3)])

def f(x, u):
  return A @ x + B @ u

def integrate(dt):
  def concat(Ak, Bk_0, Bk_1):
    return np.hstack([Ak.flatten(), Bk_0.flatten(), Bk_1.flatten()])

  def split(res):
    res_split = np.split(res, np.cumsum([N_X*N_X, N_X*N_U, N_X*N_U])[:-1])
    Ak = res_split[0].reshape((N_X, N_X))
    Bk_0 = res_split[1].reshape((N_X, N_U))
    Bk_1 = res_split[2].reshape((N_X, N_U))

    return Ak, Bk_0, Bk_1

  def res_dot(t, res):
    Ak, Bk_0, Bk_1 = split(res)

    Ak_dot = A @ Ak
    Bk_0_dot = A @ Bk_0 + B * (dt - t) / dt
    Bk_1_dot = A @ Bk_1 + B * t / dt

    return concat(Ak_dot, Bk_0_dot, Bk_1_dot)

  Ak_i = np.eye(N_X)
  Bk_0_i = np.zeros((N_X, N_U))
  Bk_1_i = np.zeros((N_X, N_U))
  res_i = concat(Ak_i, Bk_0_i, Bk_1_i)
  res = solve_ivp(res_dot, (0.0, dt), res_i).y[:, -1]
  Ak, Bk_0, Bk_1 = split(res)

  return Ak, Bk_0, Bk_1

def solve(rho):
  x = cp.Variable((N + 1, N_X))
  u = cp.Variable((N + 1, N_U))
  sigma = cp.Variable(N + 1)

  cost = 0.0
  constr = []

  cost += 100 * cp.norm2(x[N] - X_F)
  cost += cp.norm2(sigma)

  # dynamics constraints
  Ak, Bk_0, Bk_1 = integrate(T_F / N)
  for i in range(N):
    constr += [x[i + 1] == Ak @ x[i] + Bk_0 @ u[i] + Bk_1 @ u[i + 1]]

  # control constraints
  for i in range(N + 1):
    constr += [rho <= sigma[i], sigma[i] <= RHO_MAX]
    constr += [cp.norm2(u[i]) <= sigma[i]]
  for i in range(N):
    constr += [cp.norm2(u[i + 1] - u[i]) <= 2 * np.sqrt(rho**2 - RHO_MIN**2)]

  # initial conditions
  constr += [x[0] == X_I]

  prob = cp.Problem(cp.Minimize(cost), constr)
  cost = prob.solve(solver=cp.CLARABEL)

  eta_N = constr[N - 1].dual_value

  return cost, x.value, u.value, eta_N

def fohlcvx():
  rho_low = RHO_MIN
  rho_high = RHO_MAX
  while rho_high - rho_low > CONV_EPS:
    rho_1 = rho_low + (rho_high - rho_low) / 3
    rho_2 = rho_high - (rho_high - rho_low) / 3
    cost_1, x_1, u_1, eta_1_N = solve(rho_1)
    cost_2, x_2, u_2, eta_2_N = solve(rho_2)
    if not np.all(np.abs(eta_1_N) <= VIOL_EPS) and np.sum(np.linalg.norm(u_1, axis=1) < RHO_MIN - VIOL_EPS) > N_X + 1:
      rho_low = rho_1
    elif np.all(np.abs(eta_2_N) <= VIOL_EPS):
      rho_high = rho_2
    else:
      if cost_1 > cost_2:
        rho_low = rho_1
      else:
        rho_high = rho_2
  rho = (rho_low + rho_high) / 2
  cost, x, u, eta_N = solve(rho)
  return cost, x, u, rho

if __name__ == '__main__':
  cost, x, u, rho = fohlcvx()
  print(f'cost: {cost}')
  print(f'rho: {rho}')
  plot(x, u, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, True)
