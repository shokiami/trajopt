from viz import plot
import numpy as np
from foh import solve, f, N, X_I, T_F, RHO_MIN, RHO_MAX, N_X, VIOL_EPS
import matplotlib.pyplot as plt

PLOT_RHOS = [4.025, 5.5]

if __name__ == '__main__':
  rhos = np.round(np.arange(RHO_MIN, RHO_MAX + 0.001, 0.01), 3)
  red = []
  blue = []
  orange = []
  costs = []
  min_cost = np.inf
  min_rho = 0
  for i in range(len(rhos)):
    cost, x, u, eta_N = solve(rhos[i])
    if not np.all(np.abs(eta_N) <= VIOL_EPS) and np.sum(np.linalg.norm(u, axis=1) < RHO_MIN - VIOL_EPS) > N_X + 1:
      red.append(i)
    elif np.all(np.abs(eta_N) <= VIOL_EPS):
      orange.append(i)
    else:
      blue.append(i)
      if cost <= min_cost:
        min_rho = rhos[i]
        min_cost = cost
    costs.append(cost)
    print(cost, rhos[i])

    # if rhos[i] in PLOT_RHOS:
    #   plot(x, u, f, X_I, np.full(N, T_F / N), RHO_MIN, RHO_MAX, True, r'$\tilde{\rho}_{\min} = $' + str(rhos[i]))

  fig = plt.figure()
  ax = fig.add_subplot()
  ax.set_title(r'Cost vs $\tilde{\rho}_{\min}$')
  ax.set_ylim(15, 45)

  costs = np.array(costs)
  red = red + [blue[0]]
  orange = [blue[-1]] + orange
  ax.plot(rhos[red], costs[red], linestyle='-', c='red', label=r"LCvx violated at $> n_x + 1$ vertices")
  ax.plot(rhos[blue], costs[blue], linestyle='-', c='blue', label=r"$\tilde{\rho}_{\min} \in (\rho_{\min}', \rho_{\min})$")
  ax.plot(rhos[orange], costs[orange], linestyle='-', c='orange', label=r"$\eta_N = 0$")

  ax.axvline(RHO_MIN, color='blue', linestyle='--', alpha=0.5, label=r"$\rho_{\min}, \rho_{\max}$")
  ax.axvline(RHO_MAX, color='blue', linestyle='--', alpha=0.5)
  ax.axvline(rhos[blue[0]], color='green', linestyle='--', alpha=0.5, label=r"$\rho_{\min}^-, \rho_{\min}^+$")
  ax.axvline(rhos[blue[-1]], color='green', linestyle='--', alpha=0.5)
  ax.plot(min_rho, min_cost, 'o', c='purple', label='opt')

  ax.set_xlabel(r'$\tilde{\rho}_{\min}$')
  ax.set_ylabel('Cost')
  ax.legend(loc='upper right')
  ax.grid()
  plt.show()
