from viz import plot
import cvxpy as cp
import numpy as np
from scipy.integrate import solve_ivp

N = 10
X_I = [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
X_F = [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]
T_F = 4.0
U_MIN = 4.0
U_MAX = 6.0

DU_MAX = 2.5

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

def solve(u_min, u_max):
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
    constr += [u_min <= sigma[i], sigma[i] <= u_max]
  for i in range(N):
    constr += [cp.norm2(u[i + 1] - u[i]) <= DU_MAX]

  # initial conditions
  constr += [x[0] == X_I]

  # final conditions
  constr += [x[N] == X_F]

  prob = cp.Problem(cp.Minimize(cost), constr)
  cost = prob.solve(solver=cp.CLARABEL)

  return cost, x.value, u.value


if __name__ == '__main__':

  # cost, x, u = solve(U_MIN, U_MAX)
  # max_theta = 0.0
  # max_d = 0.0
  # for i in range(len(u) - 1):
  #   theta = np.arccos(np.dot(u[i], u[i + 1]) / (np.linalg.norm(u[i]) * np.linalg.norm(u[i + 1])))
  #   d = np.linalg.norm(u[i + 1] - u[i])
  #   if theta > max_theta:
  #     max_theta = theta
  #   if d > max_d:
  #     max_d = d
  # print(max_theta, max_d)

  theta = np.arccos(1.0 - DU_MAX**2 / (2 * U_MIN**2))
  print(theta)
  
  u_min = U_MIN / np.cos(theta / 2)

  cost, x, u = solve(u_min, U_MAX)
  print(f'cost: {cost}')
  
  plot(x, u, f, X_I, np.full(N, T_F / N), U_MIN, U_MAX, True)
