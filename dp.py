import math
import numpy as np
from map import Road

def newton(p_rider, velocity, k_wheel, C_d, rho_air, area, M, mu, theta, F_b):
    return (k_wheel * p_rider / velocity - 1 / 2 * C_d * rho_air * area * (velocity ** 2) - M * 9.8 *(mu * math.cos(theta) + math.sin(theta)) - F_b) / M

def get_index(x,delta_x):
    return int(x / delta_x)
def dp(roads,k_wheel, C_d, rho_air, area, M, mu, theta, F_b,k_f,k_r,k_i,p_max0):
    nx,nv,ns,ne = 10,40,10,10
    dx,dv,ds,de = 100,0.5,100,1000
    max_x_n = 10
    delta_p = 10
    dp = np.ones((nx,nv,ns,ne)) * 100000
    dicision = [np.zeros((nx,nv,ns,ne)),np.zeros((nx,nv,ns,ne)),np.zeros((nx,nv,ns,ne)),np.zeros((nx,nv,ns,ne))]
    dp[0,:,ns-1,ne-1] = 0
    #print(dp
    p_x = np.zeros((nx))
    for x_i in range(1,nx):
        res = 100000
        for x_j in range(x_i - max_x_n,x_i):
            if x_j >= 0:
                for v_j in range(1,nv):
                    for s_j in range(ns):
                        for e_j in range(ne):
                            p_max = s_j * ds
                            for p_rider in np.arange(0,p_max,delta_p):
                                dt = (x_i - x_j) * dx / (v_j * dv)
                                v_i = get_index(v_j * dv + newton(p_rider,v_j,k_wheel,C_d,rho_air,area,M,mu,theta,F_b) * dt,dv)
                                #print(newton(p_rider,v_j,k_wheel,C_d,rho_air,area,M,mu,theta,F_b))
                                delta_p_max = - k_f * p_max * p_rider / p_max0 + k_r * (p_max0 - p_max)
                                s_i = get_index(s_j * ds + delta_p_max * dt,ds)
                                e_i = get_index(e_j * de + k_i * (p_max0 - p_max) * dt - p_rider * dt,de)
                                #print(x_i,v_i,s_i,e_i)
                                if v_i < nv and s_i < ns and e_i < ne and v_i > 0 and s_i > 0 and e_i > 0:
                                    if dp[x_j,v_j,s_j,e_j] + dt < dp[x_i,v_i,s_i,e_i]:
                                        dp[x_i,v_i,s_i,e_i] = dp[x_j,v_j,s_j,e_j] + dt
                                        dicision[0][x_i,v_i,s_i,e_i] = x_j
                                        dicision[1][x_i,v_i,s_i,e_i] = v_j
                                        dicision[2][x_i,v_i,s_i,e_i] = s_j
                                        dicision[3][x_i,v_i,s_i,e_i] = e_j
                                        #print(x_j,v_j,s_j,e_j,dp[x_j,v_j,s_j,e_j] + dt)
                                        #print(x_i,v_i,s_i,e_i,dp[x_i,v_i,s_i,e_i])
                                        res = min(res,dp[x_i,v_i,s_i,e_i])
        print(x_i,res)
        
    return dp

if __name__ == '__main__':
    dp(0,k_wheel=1,C_d=0.05,rho_air=1.1,area=0.35,M=60,mu=0.2,theta=0,F_b=0,k_f=0.036,k_r=0.0124,k_i=0.556,p_max0=1150)