import numpy as np
import matplotlib.pyplot as plt

def load_data(fpath):
    data = np.loadtxt(fpath)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    return x, y, z

def get_coef(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum):
    A = np.array([
    [xx, xy, xz, x_sum],
    [xy, yy, yz, y_sum],
    [xz, yz, zz, z_sum],
    [x_sum, y_sum, z_sum, N]
    ])
    coef = np.linalg.solve(A, np.zeros(4))
    a, b, c, d = coef[0], coef[1], coef[2], coef[3]
    return a, b, c, d

def get_svd(A, b):
    A = np.array(A)
    U, S, VH = np.linalg.svd(A, full_matrices=False)
    S1 = 1/S
    S1 = np.diag(S1)
    S1[np.abs(S) < 1e-8] = 0
    x = VH.T @ S1 @ U.T @ b
    return x

def get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum):
    A = np.array([
    [xx, xy, x_sum],
    [xy, yy, y_sum],
    [xz, yz, z_sum],
    [x_sum, y_sum, N]
    ])
    # let c = 1
    q = np.array([-xz, -yz, -zz, -z_sum])
    coefs = get_svd(A.T @ A, A.T @ q)
    return coefs[0], coefs[1], 1, coefs[2]

def get_fx(a, b, c, d, X, Y):
    return -(a*X + b*Y + d) / c

def plot(a, b, c, d, x, y, z):
    ax = plt.axes(projection='3d')
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z,  color='g')

    # x, y = np.linspace(-1, 1, 10), np.linspace(0.2, 0.5, 10)
    x = np.linspace(min(x),max(x),3)
    y = np.linspace(min(y),max(y),3)
    X, Y = np.meshgrid(x,y)
    Z = get_fx(a, b, c, d, X, Y)
    ax.plot_surface(X, Y, Z, alpha=0.5, color='b')
    plt.show()


if __name__ == "__main__": 
    # # 4a.
    # x, y, z = load_data("clear_table.txt")
    # 4b.
    x, y, z = load_data("cluttered_table.txt")
    N = x.shape[0]
    xx = np.sum(x*x)
    yy = np.sum(y*y)
    zz = np.sum(z*z)
    xy = np.sum(x*y)
    xz = np.sum(x*z)
    yz = np.sum(y*z)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    z_sum = np.sum(z)

    # degeneracies
    a, b, c, d = get_coef(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)
    print(f"degeneracies: {a}, {b}, {c}, {d}")

    # avoid degeneracies: e.g. let c = 1 but others vary
    a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)
    print(f"coefficients: {a}, {b}, {c}, {d}")

    plot(a, b, c, d, x, y, z)

    avg_dist = np.mean(np.abs((a*x + b*y + c*z + d)) / np.sqrt(a**2 + b**2 + c**2))
    print(f"average distance is {avg_dist}")



    

