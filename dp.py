import math
import numpy as np
from map import Road,get_roads
import matplotlib.pyplot as plt

def newton(p_rider, velocity, k_wheel, C_d, rho_air, area, M, mu, theta, F_b):
    return (k_wheel * p_rider / velocity - 1 / 2 * C_d * rho_air * area * (velocity ** 2) - M * 9.8 *(mu * math.cos(theta) + math.sin(theta)) - F_b) / M

def get_index(x,delta_x):
    return int(x / delta_x)

def lower_bounds(nums,target):
    l,r = 0,len(nums)-1
    res = len(nums)
    while l < r:
        mid = (l + r) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    if nums[l] >= target:
        res = l
    return res

def dp(roads,k_wheel, C_d, rho_air, area, M, mu, theta, F_b,k_f,k_r,k_i,p_max0):
    
    total_dis = []
    for road in roads:
        if len(total_dis) == 0:
            total_dis.append(road.road_length)
        else:
            total_dis.append(total_dis[len(total_dis)-1] + road.road_length)

    print(total_dis[len(total_dis)-1])

    nx,nv,ns,ne = 424,30,10,10
    dx,dv,ds,de = 50,0.5,150,1000
    max_x_n = 3
    delta_p = 10
    dp = np.ones((nx,nv,ns,ne)) * 100000
    dicision = np.zeros((nx,nv,ns,ne,5))
    dp[0,0:10,ns-1,ne-1] = 0

    p_x = np.zeros((nx))
    v_x = np.zeros((nx))

    for x_i in range(1,nx):
        res = 100000
        for x_j in range(x_i - max_x_n,x_i):
            if x_j >= 0:
                road_id = lower_bounds(total_dis,x_j * dx)
                theta = roads[road_id].theta
                rho = roads[road_id].rho
                #print(x_j,road_id,theta,rho)
                for v_j in range(1,nv):
                    for s_j in range(1,ns):
                        for e_j in range(1,ne):
                            p_max = s_j * ds
                            for p_rider in np.arange(0,p_max,delta_p):
                                dt = (x_i - x_j) * dx / (v_j * dv)
                                v_i = get_index(v_j * dv + newton(p_rider,v_j,k_wheel,C_d,rho_air,area,M,mu,theta,0) * dt,dv)
                                #print(newton(p_rider,v_j,k_wheel,C_d,rho_air,area,M,mu,theta,F_b))
                                delta_p_max = - k_f * p_max * p_rider / p_max0 + k_r * (p_max0 - p_max)
                                s_i = get_index(s_j * ds + delta_p_max * dt,ds)
                                e_i = get_index(e_j * de + k_i * (p_max0 - p_max) * dt - p_rider * dt,de)

                                if v_i < nv and s_i < ns and e_i < ne and v_i > 0 and s_i > 0 and e_i > 0:
                                    if dp[x_j,v_j,s_j,e_j] + dt < dp[x_i,v_i,s_i,e_i]:
                                        dp[x_i,v_i,s_i,e_i] = dp[x_j,v_j,s_j,e_j] + dt
                                        dicision[x_i,v_i,s_i,e_i] = x_j,v_j,s_j,e_j,p_rider
                                        #print(x_j,v_j,s_j,e_j,dp[x_j,v_j,s_j,e_j] + dt)
                                        #print(x_i,v_i,s_i,e_i,dp[x_i,v_i,s_i,e_i])
                                        res = min(res,dp[x_i,v_i,s_i,e_i])
                                
                                v_i = get_index(v_j * dv + newton(p_rider,v_j,k_wheel,C_d,rho_air,area,M,mu,theta,F_b) * dt,dv)
                                delta_p_max = - k_f * p_max * p_rider / p_max0 + k_r * (p_max0 - p_max)
                                s_i = get_index(s_j * ds + delta_p_max * dt,ds)
                                e_i = get_index(e_j * de + k_i * (p_max0 - p_max) * dt - p_rider * dt,de)
                                if v_i < nv and s_i < ns and e_i < ne and v_i > 0 and s_i > 0 and e_i > 0:
                                    if dp[x_j,v_j,s_j,e_j] + dt < dp[x_i,v_i,s_i,e_i]:
                                        dp[x_i,v_i,s_i,e_i] = dp[x_j,v_j,s_j,e_j] + dt
                                        dicision[x_i,v_i,s_i,e_i] = x_j,v_j,s_j,e_j,p_rider
                                        #print(x_j,v_j,s_j,e_j,dp[x_j,v_j,s_j,e_j] + dt)
                                        #print(x_i,v_i,s_i,e_i,dp[x_i,v_i,s_i,e_i])
                                        res = min(res,dp[x_i,v_i,s_i,e_i])
        print(x_i,res)
    ans = 100000
    x,v,s,e = nx-1,0,0,0
    for v_i in range(1,nv):
        for s_i in range(ns):
            for e_i in range(ne):
                if dp[nx-1,v_i,s_i,e_i] < ans:
                    ans = dp[nx-1,v_i,s_i,e_i]
                    v,s,e = v_i,s_i,e_i
    while x > 0:
        x_i,v_i,s_i,e_i,p_i = map(int,dicision[x,v,s,e])
        p_i = dicision[x,v,s,e,4]
        for x_j in range(x_i,x+1):
            p_x[x_j] = p_i
        for x_j in range(x_i,x+1):
            v_x[x_j] = v_i
        x,v,s,e = x_i,v_i,s_i,e_i
    print(p_x)
    plt.plot(np.arange(0,nx*dx,dx),p_x)
    plt.show()
    return dp

if __name__ == '__main__':
    roads = get_roads()
    dp(roads,k_wheel=1,C_d=0.05,rho_air=1.1,area=0.35,M=60,mu=0.003,theta=0.001,F_b=40,k_f=0.036,k_r=0.0124,k_i=0.556,p_max0=1150)