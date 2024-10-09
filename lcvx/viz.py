import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def draw_sphere(ax, x0, y0, z0, r):
  u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
  x = r * np.cos(u) * np.sin(v) + x0
  y = r * np.sin(u) * np.sin(v) + y0
  z = r * np.cos(v) + z0
  ax.plot_surface(x, y, z, color='blue', alpha=0.2)

def single_shot(f, x_i, u, T, foh):
  T_cumsum = np.cumsum(T)

  def x_dot(t, x):
    i = np.searchsorted(T_cumsum, t)
    tau = (t - (T_cumsum[i - 1] if i > 0 else 0.0)) / T[i]
    if foh:
      u_foh = (1.0 - tau) * u[i] + tau * u[i + 1]
      return f(x, u_foh)
    else:
      return f(x, u[i])
  
  res = solve_ivp(x_dot, (0.0, T_cumsum[-1]), x_i, max_step=1e-2).y[:]
  r_prop = res[:3]
  return r_prop

def plot(x, u, f, x_i, T, u_min, u_max, foh = False):
  fig = plt.figure('trajopt', figsize=(12, 4))
  r_x, r_y, r_z = x[:, :3].swapaxes(0, 1)
  u_x, u_y, u_z = u.swapaxes(0, 1)
  r_prop_x, r_prop_y, r_prop_z = single_shot(f, x_i, u, T, foh)

  # plot traj
  ax1 = fig.add_subplot(131, projection='3d')
  ax1.set_title('Optimal Trajectory')

  # plot controls
  if foh:
    ax1.quiver(r_x, r_y, r_z, 0.1 * u_x, 0.1 * u_y, 0.1 * u_z, color='red')
  else:
    ax1.quiver(r_x[:len(u)], r_y[:len(u)], r_z[:len(u)], 0.1 * u_x, 0.1 * u_y, 0.1 * u_z, color='red')

  # plot traj
  ax1.plot(r_prop_x, r_prop_y, r_prop_z)

  # plot start and dest
  ax1.plot(r_x[0], r_y[0], r_z[0], 'o', color='green', label='start')
  ax1.plot(r_x[-1], r_y[-1], r_z[-1], 'o', color='purple', label='dest')

  ax1.set_xlim(-1.0, 11.0)
  ax1.set_ylim(-1.0, 11.0)
  ax1.set_zlim(-1.0, 11.0)
  ax1.set_aspect('equal')
  ax1.legend()

  ax2 = fig.add_subplot(132)
  ax2.set_title('Control Magnitude')
  ax2.axhline(u_max, color='red', linestyle='--', alpha=0.5)
  ax2.axhline(u_min, color='red', linestyle='--', alpha=0.5)
  if foh:
    step = 0.01
    k = np.arange(0, len(x) - 1 + step, step)
    u_interp = np.array([np.interp(k, range(len(x)), u[:, i]) for i in range(3)]).swapaxes(0, 1)
    u_norm = np.linalg.norm(u_interp, axis=1)
    ax2.plot(k, u_norm)
  else:
    k = range(len(x))
    u_norm = np.linalg.norm(u, axis=1)
    ax2.step(k, np.append(u_norm, u_norm[-1]), where='post')
  ax2.grid()
  ax2.set_xticks(range(len(x)))
  ax2.set_xlim(0, len(x) - 1)
  print(f'u_min: {np.min(u_norm)}, u_max: {np.max(u_norm)}')

  ax3 = fig.add_subplot(133, projection='3d')
  draw_sphere(ax3, 0.0, 0.0, 0.0, u_min)
  draw_sphere(ax3, 0.0, 0.0, 0.0, u_max)
  if foh:
    ax3.plot(u_x, u_y, u_z)
  else:
    ax3.scatter(u_x, u_y, u_z, s=5)
  ax3.set_aspect('equal')
  
  plt.tight_layout()
  plt.show()
