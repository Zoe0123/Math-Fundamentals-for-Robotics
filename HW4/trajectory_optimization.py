import numpy as np
# from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage import distance_transform_edt
# import scipy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

waypoints = 300
N = 101
OBST = np.array([[20, 30], [60, 40], [70, 85]])
epsilon = np.array([[25], [20], [30]])

obs_cost = np.zeros((N, N))
for i in range(OBST.shape[0]):
    t = np.ones((N, N))
    t[OBST[i, 0], OBST[i, 1]] = 0
    t_cost = distance_transform_edt(t)
    t_cost[t_cost > epsilon[i]] = epsilon[i]
    t_cost = 1 / (2 * epsilon[i]) * (t_cost - epsilon[i])**2
    obs_cost = obs_cost + t_cost

gx, gy = np.gradient(obs_cost)

SX = 10
SY = 10
GX = 90
GY = 90

traj = np.zeros((2, waypoints))
traj[0, 0] = SX
traj[1, 0] = SY
dist_x = GX-SX
dist_y = GY-SY
for i in range(1, waypoints):
    traj[0, i] = traj[0, i-1] + dist_x/(waypoints-1)
    traj[1, i] = traj[1, i-1] + dist_y/(waypoints-1)

path_init = traj.T
tt = path_init.shape[0]
path_init_values = np.zeros((tt, 1))
for i in range(tt):
    path_init_values[i] = obs_cost[int(np.floor(path_init[i, 0])), int(np.floor(path_init[i, 1]))]

# # Plot 2D
# plt.imshow(obs_cost.T)
# plt.plot(path_init[:, 0], path_init[:, 1], 'ro')

# # Plot 3D
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(111, projection='3d')
# xx, yy = np.meshgrid(range(N), range(N))
# print(xx.shape)
# print(obs_cost.T.shape)
# ax3d.plot_surface(xx, yy, obs_cost.T, cmap=plt.get_cmap('coolwarm'))
# ax3d.scatter(path_init[:, 0], path_init[:, 1], path_init_values, s=20, c='r')

# plt.show()

path = path_init

# # 6a.
# # after 1 iteration
# n = path.shape[0]
# for i in range(1, n-1):
#     x, y = int(np.floor(path[i,0])), int(np.floor(path[i,1]))
#     path[i,0] -= 0.1*gx[x, y]
#     path[i,1] -= 0.1*gy[x, y]

# # after convergence
# n = path.shape[0]
# diff = 100
# # convergence criteria
# while diff > 1e-2:
#     curr_dist = np.linalg.norm(path)
#     for i in range(1, n-1):
#         x, y = int(np.floor(path[i,0])), int(np.floor(path[i,1]))
#         path[i,0] -= 0.1*gx[x, y]
#         path[i,1] -= 0.1*gy[x, y]
#     diff = abs(curr_dist-np.linalg.norm(path))


# # 6b.
# # 100, 200, 500
# n = path.shape[0]
# for iter in range(500):
#     for i in range(1, n-1):
#         x, y = int(np.floor(path[i,0])), int(np.floor(path[i,1]))
#         g_x = 0.8*gx[x, y] + 4*(path[i,0] - path[i-1,0])
#         g_y = 0.8*gy[x, y] + 4*(path[i,1] - path[i-1,1])
#         path[i,0] -= 0.1*g_x
#         path[i,1] -= 0.1*g_y 


# # 6c.
# # 100, 5000
n = path.shape[0]
for iter in range(5000):
    for i in range(1, n-1):
        x, y = int(np.floor(path[i,0])), int(np.floor(path[i,1]))
        g_x = 0.8*gx[x,y] + 4*(2*path[i,0] - path[i-1,0] - path[i+1,0])
        g_y = 0.8*gy[x,y] + 4*(2*path[i,1] - path[i-1,1] - path[i+1,1])
        path[i,0] -= 0.1*g_x
        path[i,1] -= 0.1*g_y

path_values = np.zeros((tt, 1))
for i in range(tt):
    path_values[i] = obs_cost[int(np.floor(path[i, 0])), int(np.floor(path[i, 1]))]  

plt.imshow(obs_cost.T)
plt.plot(path[:, 0], path[:, 1], 'ro')
plt.title('6(c). path after 5000 iterations')

fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(range(N), range(N))
ax3d.plot_surface(xx, yy, obs_cost.T, cmap=plt.get_cmap('coolwarm'))
ax3d.scatter(path[:, 0], path[:, 1], path_values, s=20, c='r')

plt.show()

