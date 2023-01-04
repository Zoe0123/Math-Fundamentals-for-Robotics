import numpy as np
import matplotlib.pyplot as plt
from q4_ab import get_coef1, get_fx
from q4_c import load_data, get_entries

def ransac(data, iters, thres):
    N = data.shape[0]
    maxInliers, bestInliers = -1, None
    for i in range(iters):
        inds = np.random.randint(0, N, 1)
        points = data[inds,:]
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum = get_entries(x, y, z)
        a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)
        
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        dist = np.abs((a*x + b*y + c*z + d) / np.sqrt(a**2 + b**2 + c**2))

        inliers = dist < thres
        numInliers = sum(inliers)
        
        if numInliers > maxInliers:
            maxInliers = numInliers
            bestInliers = inliers
    remain = ~bestInliers
    return data[bestInliers,:],  data[remain,:]

if __name__ == "__main__":
    data = load_data("clean_hallway.txt")
    # "clutterd_hallway.txt"
    N = data.shape[0]
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, color='g')

    bestPoints, remain = ransac(data, 2000, 0.008)
    x, y, z = bestPoints[:, 0], bestPoints[:, 1], bestPoints[:, 2]
    xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum = get_entries(x, y, z)
    a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)
    x = np.linspace(min(x),max(x),3)
    y = np.linspace(min(y),max(y),3)
    X, Y = np.meshgrid(x,y)
    Z = get_fx(a, b, c, d, X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='b')

    bestPoints, remain = ransac(remain, 2000, 0.008)
    x, y, z = bestPoints[:, 0], bestPoints[:, 1], bestPoints[:, 2]
    xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum = get_entries(x, y, z)
    a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)

    x = np.linspace(min(x),max(x),3)
    y = np.linspace(min(y),max(y),3)
    X, Y = np.meshgrid(x,y)
    Z = get_fx(a, b, c, d, X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='b')

    bestPoints, remain = ransac(remain, 2000, 0.008)
    x, y, z = bestPoints[:, 0], bestPoints[:, 1], bestPoints[:, 2]
    xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum = get_entries(x, y, z)
    a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)

    x = np.linspace(min(x),max(x),3)
    y = np.linspace(min(y),max(y),3)
    X, Y = np.meshgrid(x,y)
    Z = get_fx(a, b, c, d, X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='b')

    bestPoints, remain = ransac(remain, 2000, 0.008)
    x, y, z = bestPoints[:, 0], bestPoints[:, 1], bestPoints[:, 2]
    xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum = get_entries(x, y, z)
    a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)

    x = np.linspace(min(x),max(x),3)
    y = np.linspace(min(y),max(y),3)
    X, Y = np.meshgrid(x,y)
    Z = get_fx(a, b, c, d, X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='b')

    plt.show()
