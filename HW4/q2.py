import numpy as np
import matplotlib.pyplot as plt

def fxy(x, y):
    return np.power(x,3) + np.power(y,3) - 9*np.power(x,2) + np.power(y,2) + 7

# def deriv_x(x):
#     return 3 * (x**2) - 18 * x

# def deriv_y(y):
#     return 3 * (y**2) + 2 * y

if __name__ == "__main__":
    fig,ax = plt.subplots()
    x = np.linspace(-2,8,300)
    y = np.linspace(-2,8,300)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    x,y = np.meshgrid(x,y)
    f = fxy(x, y)
    # ax.contourf(x, y, deriv_x(x), colors='b')
    plt.contourf(x, y, f, 20, cmap='RdGy')
    plt.plot([6, 0, 0, 6],[0, -2/3, 0, -2/3],'ko')
    plt.show()
