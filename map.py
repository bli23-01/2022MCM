import code
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Road:
    def __init__(self, start_point, end_point):
        self.start_point = start_point #标准化之后的点信息
        self.end_point = end_point
        self.height = end_point[2] - start_point[2] 
        self.project_length = math.sqrt((start_point[0] - end_point[0]) ** 2 + 
                                        (start_point[1] - end_point[1]) ** 2)
        self.road_length = math.sqrt(self.project_length ** 2 + 
                                    self.height ** 2)
        self.theta = math.atan(self.height / self.project_length)

class Rider:
    def __init__(self, gender = True):
        self.gender = gender
        self.velocity = 2
        self.cur_road = 0 #the index of roads
        self.cur_point = roads[self.cur_road].start_point
    def update(self, rate = 1):    # 1s
        print("update")
        self.cur_road = (self.cur_road + 1) % len(roads)
        self.cur_point = roads[self.cur_road].start_point
        return self.cur_point
        
def update(i):
    #ax.scatter3D(coods[:, 0],coods[:, 1], coods[:, 2], cmap='Blues')  #绘制散点图
    #plt.clear()
    ax.plot3D(coods[:, 0],coods[:, 1], coods[:, 2], 'gray')
    riders_points = []
    for rider in riders:
        riders_points.append(rider.update())
    riders_points = np.array(riders_points)
    ax.scatter3D(riders_points[:, 0], riders_points[:, 1], riders_points[:, 2], cmap='Blues')

    
if __name__ == '__main__':
    f = open("data.csv", "r")
    s = f.read()
    coodStrs = s.split(" ")
    coods = []
    zero_cood = list(map(float,coodStrs[0].split(",")))
    for s1 in coodStrs:
        t = list(map(float,s1.split(",")))
        t[0] = (t[0] - zero_cood[0]) / 0.00001141
        t[1] = (t[1] - zero_cood[1]) / 0.00000899
        coods.append(t)

    global roads
    roads = []

    for i in range(0, len(coods)):
        roads.append(Road(coods[i], coods[(i + 1) % len(coods)]))
        print(roads[i].road_length)

    coods = np.array(coods)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plt.axis('auto')
    ax.set_xlim([-3000,2000])
    ax.set_ylim([-4000,1000])
    ax.set_zlim([0,1000])

    global riders 
    riders = [Rider()]

    ax.plot3D(coods[:, 0],coods[:, 1], coods[:, 2], 'gray')
    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=1)
    plt.show()

    f.close()
