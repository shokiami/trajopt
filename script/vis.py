import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

df = pd.read_csv('data/traj.txt', sep='\s+')
r_x = df['r_x'].to_numpy(dtype=float)
r_y = df['r_y'].to_numpy(dtype=float)
r_z = df['r_z'].to_numpy(dtype=float)
u_x = df['u_x'].to_numpy(dtype=float)
u_y = df['u_y'].to_numpy(dtype=float)
u_z = df['u_z'].to_numpy(dtype=float)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(r_x, r_y, r_z)
for i in range(len(r_x)):
  ax.quiver(r_x, r_y, r_z, 0.1 * u_x, 0.1 * u_y, 0.1 * u_z, color='red')
plt.show()
