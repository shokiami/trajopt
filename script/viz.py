import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

U_SCALAR = 0.1

df = pd.read_csv('data/traj.csv')
r_x = df['r_x'].to_numpy(dtype=float)
r_y = df['r_y'].to_numpy(dtype=float)
r_z = df['r_z'].to_numpy(dtype=float)
u_x = df['u_x'].to_numpy(dtype=float)
u_y = df['u_y'].to_numpy(dtype=float)
u_z = df['u_z'].to_numpy(dtype=float)

ax = plt.figure('Optimal Trajectory').add_subplot(projection='3d')
ax.plot(r_x, r_y, r_z)
for i in range(len(r_x)):
  ax.quiver(r_x, r_y, r_z, U_SCALAR * u_x, U_SCALAR * u_y, U_SCALAR * u_z, color='red')
plt.show()
