import numpy as np
import matplotlib.pyplot as plt

def load_data(fpath):
    f_x = np.loadtxt(fpath)
    return f_x

def get_coeff(X, f_X):
    A = [np.cos(3*np.pi*X), X, np.ones(101)]
    # use SVD to find the coefficients 
    u, s, vh = np.linalg.svd(np.array(A).T)
    s1 = np.zeros((101, 3))
    for i in range(3):
        if s[i] != 0:
            s1[i][i] = 1/s[i]
        else:
            print("no solution")
            return np.array([-1, -1, -1])
    return (vh.T@s1.T)@u.T@f_X
    
def get_px(a, b, c, x):
    return a*np.cos(3*np.pi*x) + b*x + c

if __name__ == '__main__':
    f_X = load_data('problem2.txt')
    # # original graph
    # plt.figure(figsize=(10, 8))
    # x = np.linspace(0, 1, num=101)
    # plt.xlabel('x')
    # plt.ylabel('f_x')
    # plt.plot(x, f_x, '-ko')
    # plt.show()  
    X = np.linspace(0, 1, num=101)
    coeff = get_coeff(X, f_X)
    print(coeff)
    a, b, c = coeff[0], coeff[1], coeff[2]
    plt.figure(figsize=(10, 8))
    plt.xlabel('x')
    plt.ylabel('p_x')
    plt.plot(X, f_X, 'ko')
    plt.plot(X, get_px(a, b, c, X), '-r')
    plt.legend(['black points: f_x points', 'red line: estimated p(x) in line'])
    plt.show()

    