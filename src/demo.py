from viz import plot
import numpy as np
from foh import solve, f, N, X_I, T_F, RHO_MIN, RHO_MAX, N_X, VIOL_EPS
import matplotlib.pyplot as plt

PLOT_RHOS = [4.05, 5.5]

if __name__ == '__main__':
  rhos = np.round(np.arange(RHO_MIN, RHO_MAX + 0.01, 0.05), 2)
  red = []
  blue = []
  orange = []
  costs = []
  for i in range(len(rhos)):
    cost, x, u, eta_N = solve(rhos[i])
    if not np.all(np.abs(eta_N) <= 1e-6) and np.sum(np.linalg.norm(u, axis=1) < RHO_MIN - VIOL_EPS) > N_X + 1:
      red.append(i)
    elif np.all(np.abs(eta_N) <= 1e-6):
      orange.append(i)
    else:
      blue.append(i)
    costs.append(cost)
    print(cost, rhos[i])

    if rhos[i] in PLOT_RHOS:
      plot(x, u, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, True, r'$\tilde{\rho}_{\min} = $' + str(rhos[i]))

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.set_ylim(0, 100)
  ax.set_title(r'Cost vs $\tilde{\rho}_{\min}$')

  costs = np.array(costs)
  red.append(blue[0])
  orange.insert(0, blue[-1])
  ax.plot(rhos[red], costs[red], linestyle='-', c='red', label=r"$\tilde{\rho}_{\min} \leq \rho_{\min}'$")
  ax.plot(rhos[blue], costs[blue], linestyle='-', c='blue', label=r"$\tilde{\rho}_{\min} \in (\rho_{\min}', \rho_{\min})$")
  ax.plot(rhos[orange], costs[orange], linestyle='-', c='orange', label=r"$\tilde{\rho}_{\min} \geq \rho_{\max}'$")

  ax.set_xlabel(r'$\tilde{\rho}_{\min}$')
  ax.set_ylabel('Cost')
  ax.set_yticks(range(0, 101, 10))
  ax.legend()
  ax.grid()
  plt.show()
