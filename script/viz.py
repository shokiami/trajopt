import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VEC_SCALAR = 0.1

r_x, r_y, r_z, u_x, u_y, u_z = pd.read_csv('data/traj.csv').to_numpy(dtype=float).swapaxes(0, 1)

ax = plt.figure('trajopt').add_subplot(projection='3d')
ax.set_title('Optimal Trajectory')
ax.plot(r_x, r_y, r_z)
for i in range(len(r_x)):
  ax.quiver(r_x, r_y, r_z, VEC_SCALAR * u_x, VEC_SCALAR * u_y, VEC_SCALAR * u_z, color='red')
ax.plot(r_x[0], r_y[0], r_z[0], 'o', linestyle='', color='purple', label='start')
ax.plot(r_x[-1], r_y[-1], r_z[-1], 'o', linestyle='', color='green', label='dest')
ax.legend()
plt.show()
