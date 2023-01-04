import numpy as np

def get_fx(x):
    return x - np.tan(x)

def get_d_fx(x):
    return 1 - (1/np.cos(x))**2

def newton(x, thres):
    fx = get_fx(x)
    d_fx = get_d_fx(x)
    curr = fx / d_fx
    while abs(curr) >= thres:
        curr = fx / d_fx
        x -= curr
    return x

if __name__ == '__main__':
    x_lows = np.arange(13.5, 14.5, 0.05)
    for i in x_lows:
        x_low = newton(i, 0.00001)
    x_highs = np.arange(16.5, 17.5, 0.05)
    for j in x_highs:
        x_high = newton(j, 0.00001)
    print(f"x_low: {x_low}")
    print(f"x_high: {x_high}")