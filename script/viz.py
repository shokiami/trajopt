import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VEC_SCALAR = 0.1

def draw_sphere(ax, x0, y0, z0, r):
  u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
  x = r * np.cos(u) * np.sin(v) + x0
  y = r * np.sin(u) * np.sin(v) + y0
  z = r * np.cos(v) + z0
  ax.plot_surface(x, y, z, color='blue', alpha=0.5)

def plot(r, u, obs, r_prop=None):
  ax = plt.figure('trajopt').add_subplot(projection='3d')
  ax.set_title('Optimal Trajectory')

  # plot trajectory
  r_x, r_y, r_z = r.swapaxes(0, 1)
  u_x, u_y, u_z = u.swapaxes(0, 1)
  ax.plot(r_x, r_y, r_z)

  # plot controls
  for i in range(len(r)):
    ax.quiver(r_x, r_y, r_z, VEC_SCALAR * u_x, VEC_SCALAR * u_y, VEC_SCALAR * u_z, color='red')

  # plot start and dest
  ax.plot(r_x[0], r_y[0], r_z[0], 'o', linestyle='', color='green', label='start')
  ax.plot(r_x[-1], r_y[-1], r_z[-1], 'o', linestyle='', color='purple', label='dest')

  # plot obstacles
  for i in range(len(obs)):
    (obs_x, obs_y, obs_z), obs_r = obs[i]
    draw_sphere(ax, obs_x, obs_y, obs_z, obs_r)

  # plot propagated traj
  if r_prop is not None:
    r_prop_x, r_prop_y, r_prop_z = r_prop.swapaxes(0, 1)
    ax.plot(r_prop_x, r_prop_y, r_prop_z)

  ax.set_xlim(-1.0, 11.0)
  ax.set_ylim(-1.0, 11.0)
  ax.set_zlim(-1.0, 11.0)
  ax.set_aspect('equal')
  ax.legend()
  plt.show()

if __name__ == '__main__':
  traj_data = pd.read_csv('data/traj.csv').to_numpy(dtype=float)
  obs_data = pd.read_csv('data/obs.csv').to_numpy(dtype=float)
  r = traj_data[0:3]
  u = traj_data[3:6]
  obs_c = obs_data[0:3]
  obs_r = obs_data[3]
  plot(r, u, obs_c, obs_r)
