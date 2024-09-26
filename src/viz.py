import numpy as np
from scipy.integrate import solve_ivp

VEC_SCALAR = 0.1

def single_shot(f, x_i, u, T):
  T_cumsum = np.cumsum(T)

  def x_dot(t, x):
    i = np.searchsorted(T_cumsum, t)
    tau = (t - (T_cumsum[i - 1] if i > 0 else 0.0)) / T[i]
    u_foh = (1.0 - tau) * u[i] + tau * u[i + 1]
    return f(x, u_foh)
  
  res = solve_ivp(x_dot, (0.0, T_cumsum[-1]), x_i, max_step=1e-2).y[:]
  r_prop = res[:3].swapaxes(0, 1)
  return r_prop

def draw_sphere(ax, x0, y0, z0, r):
  u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
  x = r * np.cos(u) * np.sin(v) + x0
  y = r * np.sin(u) * np.sin(v) + y0
  z = r * np.cos(v) + z0
  ax.plot_surface(x, y, z, color='blue', alpha=0.5)

def plot(ax, r, u, obs, r_prop=None):
  r_x, r_y, r_z = r.swapaxes(0, 1)
  u_x, u_y, u_z = u.swapaxes(0, 1)

  # clear plot
  ax.cla()

  # plot controls
  ax.quiver(r_x, r_y, r_z, VEC_SCALAR * u_x, VEC_SCALAR * u_y, VEC_SCALAR * u_z, color='red')

  # plot start and dest
  ax.plot(r_x[0], r_y[0], r_z[0], 'o', linestyle='', color='green', label='start')
  ax.plot(r_x[-1], r_y[-1], r_z[-1], 'o', linestyle='', color='purple', label='dest')

  # plot obstacles
  for i in range(len(obs)):
    (obs_x, obs_y, obs_z), obs_r = obs[i]
    draw_sphere(ax, obs_x, obs_y, obs_z, obs_r)

  # plot traj
  if r_prop is not None:
    r_prop_x, r_prop_y, r_prop_z = r_prop.swapaxes(0, 1)
    ax.plot(r_prop_x, r_prop_y, r_prop_z)
  else:
    ax.plot(r_x, r_y, r_z)

  ax.set_xlim(-1.0, 11.0)
  ax.set_ylim(-1.0, 11.0)
  ax.set_zlim(-1.0, 11.0)
  ax.set_aspect('equal')
  ax.legend()
