from math import exp
from matplotlib import pyplot as plt
import numpy as np

def dsolve(e_0, k_f, k_r, k_i, p_max0):
    y = []
    x = []
    for p_rider in np.arange(230, 500, 0.01):
        t = dsolveP(e_0, k_f, k_r, k_i, p_max0, p_rider)
        print(t, p_rider)
        if t and t > 0.2:
            x.append(t)
            y.append(p_rider)
    plt.plot(x, y)
    plt.show()
    
def dsolveP(e_0, k_f, k_r, k_i, p_max0, p_rider):
    e = e_0
    delta_t = 0.1
    for t in np.arange(0.01, 800, delta_t):
        p_max = pmax(t, k_f, k_r, p_max0, p_rider)
        e += k_i * (p_max0 - p_max) * delta_t
        e -= p_rider * delta_t
        if p_rider > p_max or e < 0:
            return t


def pmax(t, k_f, k_r, p_max0, p_rider):
    return (p_max0 ** 2 * k_r + p_max0 * p_rider * k_f * exp(-t * (p_rider * k_f + p_max0 * k_r) / p_max0)) / (p_rider * k_f + p_max0 * k_r)

if __name__ == '__main__':
    dsolve(10000, 0.153, 0.063, 1, 500)