import numpy as np
import math

def get_divdiff(X, F_X):
    """
    Get divided difference matrix.
    """
    n = X.shape[0]
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][0] = F_X[i]
    for i in range(1, n):
        for j in range(n - i):
            matrix[j][i] = ((matrix[j][i - 1] - matrix[j + 1][i - 1]) / (X[j] - X[i + j])) 
    return matrix

def interpolate(x, X, divdiff):
    n = X.shape[0]
    P_n = divdiff[0][0] 
    for i in range(1, n):
        x_diffs = 1
        for j in range(i):
            x_diffs = x_diffs * (x - X[j])
        P_n = P_n + (x_diffs * divdiff[0][i])  
    
    return P_n

def get_c_X(n):
    X = []
    for i in range(n+1):
        X.append(i*2/n - 1)
    return X


if __name__ == '__main__':
    # q1 (b)
    X = [0, 1/32, 1/16, 3/32, 1/8]
    F_X = []
    for i in X:
        F_X.append(math.sin(2*math.pi*i))
    X, F_X = np.array(X), np.array(F_X)

    divdiff = get_divdiff(X, F_X)
    P_n = interpolate(1/10, X, divdiff)
    print(f"the result for q1 (b) is {P_n}")

    # q1 (c)
    n_examples = [2, 4, 40]
    for n in n_examples:
        X = get_c_X(n)
        F_X = []
        for i in X:
            F_X.append(1/(1+49*(i**2)))
        X, F_X = np.array(X), np.array(F_X)

        divdiff = get_divdiff(X, F_X)
        P_n = interpolate(0.09, X, divdiff)
        print(f"the interpolate result for q1 (c) for n = {n} is {P_n}")

    print(f"the actual value for f(0.09) is {1/(1+49*(0.09**2))}")

    # q1 (d)
    x = np.arange(-1, 1, 0.001)
    n_examples = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
    for n in n_examples:
        X = get_c_X(n)
        F_X = []
        for i in X:
            F_X.append(1/(1+49*(i**2)))
        X, F_X = np.array(X), np.array(F_X)

        divdiff = get_divdiff(X, F_X)

        P_ns = interpolate(x, X, divdiff)
        true_vals = 1/(1+49*(x**2))

        E_n = np.amax(np.abs(P_ns-true_vals))

        print(f"the interpolation error for q1 (d) for n = {n} is {E_n}")


    