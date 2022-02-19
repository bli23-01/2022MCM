import math

def newton(p_rider, velocity, k_wheel, C_d, rho_air, area, M, mu, theta, F_b):
    return (k_wheel * p_rider / velocity - 1 / 2 * C_d * rho_air * area * (velocity ** 2) - M * 9.8 *(mu * math.cos(theta) + math.sin(theta)) - F_b) / M

def dp():
    