import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

VEC_SCALAR = 0.1

df = pd.read_csv('data/traj.csv')
r_x = df['r_x'].to_numpy(dtype=float)
r_y = df['r_y'].to_numpy(dtype=float)
r_z = df['r_z'].to_numpy(dtype=float)
u_x = df['u_x'].to_numpy(dtype=float)
u_y = df['u_y'].to_numpy(dtype=float)
u_z = df['u_z'].to_numpy(dtype=float)

ax = plt.figure('trajopt').add_subplot(projection='3d')
ax.set_title('Optimal Trajectory')
ax.plot(r_x, r_y, r_z)
for i in range(len(r_x)):
  ax.quiver(r_x, r_y, r_z, VEC_SCALAR * u_x, VEC_SCALAR * u_y, VEC_SCALAR * u_z, color='red')
ax.plot(r_x[0], r_y[0], r_z[0], 'o', linestyle='', color='purple', label='start')
ax.plot(r_x[-1], r_y[-1], r_z[-1], 'o', linestyle='', color='green', label='dest')
ax.legend()
plt.show()
