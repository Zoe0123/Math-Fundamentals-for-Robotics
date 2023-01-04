import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fx(x):
    return np.power(x, 1/3)

def deriv_fx(y):
    return 1/(3*(y**2))

def euler(y, step):
    return y - step * deriv_fx(y)

def runge_kutta4(y, step):
    K1 = - step * deriv_fx(y)
    K2 = - step * deriv_fx(y+(K1/2))
    K3 = - step * deriv_fx(y+(K2/2))
    K4 = - step * deriv_fx(y+K3)
    return y + (K1+2*K2+2*K3+K4)/6

def adams_bashforth(y, step, f_prime):
    return y + (-step/24) * (55*f_prime[0] - 59*f_prime[1] + 37*f_prime[2] - 9*f_prime[3])

if __name__ == "__main__":
    step = 0.05
    n = int((1-0)//step)

    # 1.b
    # func = euler
    # 1.c
    # func = runge_kutta4
    # true_fx, esti_fx = np.empty(n+1), np.empty(n+1)
    # errors = np.empty(n+1)
    # true_fx[n], esti_fx[n] = 1, 1
    # errors[n] = 0
    # x = 1
    
    # for i in range(n, 0, -1):
    #     esti_fx[i-1] = func(esti_fx[i], step)
    #     x -= step
    #     true_fx[i-1] = fx(x)
    #     errors[i-1] = abs(true_fx[i-1] - esti_fx[i-1])
    
    # 1.d
    true_fx, esti_fx, f_prime = np.empty(n+4), np.empty(n+4), np.empty(n+4)
    errors = np.empty(n+1)
    true_fx[n], esti_fx[n] = 1, 1
    errors[n] = 0
    x = 1

    true_fx[n+3],true_fx[n+2],true_fx[n+1],true_fx[n] = fx(1.15), fx(1.10), fx(1.05), fx(1)
    esti_fx[n+3],esti_fx[n+2],esti_fx[n+1],esti_fx[n] = 1.04768955317165, 1.03228011545637, 1.01639635681485, 1.
    f_prime[n+3],f_prime[n+2],f_prime[n+1],f_prime[n] = deriv_fx(1.04768955317165), deriv_fx(1.03228011545637), deriv_fx(1.01639635681485), deriv_fx(1.)
    errors[n] = 0

    for i in range(n, 0, -1):
        esti_fx[i-1] = adams_bashforth(esti_fx[i], step, f_prime[i:i+4])
        f_prime[i-1] = deriv_fx(esti_fx[i-1])
        x -= step
        true_fx[i-1] = fx(x)
        errors[i-1] = abs(true_fx[i-1] - esti_fx[i-1])


    plt.figure(figsize=(10, 8))
    plt.plot(np.linspace(0, 1, n+1), true_fx[:n+1], '-b')
    plt.plot(np.linspace(0, 1, n+1), esti_fx[:n+1], '-k')
    plt.xlabel('x_i')
    plt.ylabel('true value y_xi, and estimated value y_i')
    plt.title('true value y_xi, and estimated value y_i using adams_bashforth method')
    plt.legend(['true value y_xi', 'estimated value y_i'])
    plt.savefig('1.d.png')

    d = {'x_i': np.linspace(0, 1, n+1), 'true value y(x_i)': true_fx[:n+1], 'estimated value y_i': esti_fx[:n+1], 'error abs(y(x_i)-y_i)': errors}
    df = pd.DataFrame(data=d)
    print(df)
    
    print(f'The average error is {np.sum(errors)/n}')