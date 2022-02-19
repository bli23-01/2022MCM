from math import exp
from matplotlib import pyplot as plt
import numpy as np

def dsolve(e_0, k_f, k_r, k_i, p_max0):
    y = []
    x = []
    for p_rider in np.arange(0,100,0.5):
        t = dsolveP(e_0, k_f, k_r, k_i, p_max0, p_rider)
        print(t,p_rider)
        x.append(t)
        y.append(p_rider)
    plt.plot(x,y)
    plt.show()
    
def dsolveP(e_0, k_f, k_r, k_i, p_max0, p_rider):
    e = e_0
    for t in np.arange(0.5, 400, 0.5):
        p_max = pmax(t, k_f, k_r, p_max0, p_rider)
        e += k_i * (p_max0 - p_max) * 0.5
        e -= p_rider * 0.5
        if p_rider > p_max or e < 0:
            return t


def pmax(t, k_f, k_r, p_max0, p_rider):
    return (p_max0 ** 2 * k_r + p_max0 * p_rider * k_f * exp(-t * (p_rider * k_f + p_max0 * k_r) / p_max0)) / (p_rider * k_f + p_max0 * k_r)

if __name__ == '__main__':
    dsolve(100,0.1,0.1,0.1,10)