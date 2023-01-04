import numpy as np
from q4_ab import get_coef1, plot

def load_data(fpath):
    data = np.loadtxt(fpath)
    return data

def get_entries(x, y, z):
    xx = np.sum(x*x)
    yy = np.sum(y*y)
    zz = np.sum(z*z)
    xy = np.sum(x*y)
    xz = np.sum(x*z)
    yz = np.sum(y*z)
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    z_sum = np.sum(z)
    return xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum

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
    print(maxInliers)
    return data[bestInliers,:] 

if __name__ == "__main__": 
    data = load_data("cluttered_table.txt")
    N = data.shape[0]

    bestPoints = ransac(data, 2000, 0.008)

    x, y, z = bestPoints[:, 0], bestPoints[:, 1], bestPoints[:, 2]
    print(x.shape[0])
    xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum = get_entries(x, y, z)
    a, b, c, d = get_coef1(N, xx, xy, yy, yz, xz, zz, x_sum, y_sum, z_sum)

    avg_dist = np.mean(np.abs((a*x + b*y + c*z + d)) / np.sqrt(a**2 + b**2 + c**2))
    
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    print(f"coefficients: {a}, {b}, {c}, {d}")
    print(f"average distance is {avg_dist}")
    plot(a, b, c, d, x, y, z)



