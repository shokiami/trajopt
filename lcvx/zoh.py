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

A = np.vstack([np.hstack([np.zeros((3, 3)), np.eye(3)]), np.zeros((3, 6))])
B = np.vstack([np.zeros((3, 3)), np.eye(3)])

def f(x, u):
  return A @ x + B @ u

def integrate(dt):
  def concat(Ak, Bk):
    return np.hstack([Ak.flatten(), Bk.flatten()])

  def split(res):
    Ak = res[0:36].reshape((6, 6))
    Bk_0 = res[36:54].reshape((6, 3))

    return Ak, Bk_0

  def res_dot(t, res):
    Ak, Bk_0 = split(res)

    Ak_dot = A @ Ak
    Bk_0_dot = A @ Bk_0 + B

    return concat(Ak_dot, Bk_0_dot)

  Ak_i = np.eye(6)
  Bk_0 = np.zeros((6, 3))
  res_i = concat(Ak_i, Bk_0)
  res = solve_ivp(res_dot, (0.0, dt), res_i).y[:, -1]
  Ak, Bk_0 = split(res)

  return Ak, Bk_0

if __name__ == '__main__':
  x = cp.Variable((N + 1, 6))
  u = cp.Variable((N, 3))
  sigma = cp.Variable(N)

  cost = 0.0
  constr = []

  cost += cp.norm2(sigma)

  # dynamics constraints
  Ak, Bk_0 = integrate(T_F / N)
  for i in range(N):
    constr += [x[i + 1] == Ak @ x[i] + Bk_0 @ u[i]]

  # control constraints
  for i in range(N):
    constr += [cp.norm2(u[i]) <= sigma[i]]
    constr += [cp.norm2(u[i]) <= RHO_MAX]
    constr += [RHO_MIN <= sigma[i]]

  # initial conditions
  constr += [x[0] == X_I]

  # final conditions
  constr += [x[N] == X_F]

  prob = cp.Problem(cp.Minimize(cost), constr)
  cost = prob.solve(solver=cp.CLARABEL)
  print(f'cost: {cost}')

  plot(x.value, u.value, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, False)
