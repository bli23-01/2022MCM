from dis import dis
import math
import numpy as np
from matplotlib import pyplot as plt
import datetime

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
        self.rho = 0
    def get_dir_vec_norm(self):
        v = self.end_point - self.start_point
        return v / np.sqrt(np.sum(v**2))
    @staticmethod
    def getRho(point1,point2,point3):
        a = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        b = math.sqrt((point1[0] - point3[0]) ** 2 + (point1[1] - point3[1]) ** 2)
        c = math.sqrt((point3[0] - point2[0]) ** 2 + (point3[1] - point2[1]) ** 2)
        cosA = math.sqrt(1 - ((b * b + c * c - a * a) / (2 * b * c)) ** 2)
        if cosA <= 0:
            return math.inf
        return a / (2 * cosA)
    @staticmethod
    def calcRho(roads):
        n = len(roads)
        for i in range(n):
            curRoad = roads[i]
            prevRoad = roads[(i - 1) % n]
            nextRoad = roads[(i + 1) % n]
            prevPoint = prevRoad.start_point
            nextPoint = nextRoad.end_point
            rho_start = Road.getRho(prevPoint,curRoad.start_point,nextPoint)
            rho_end = Road.getRho(prevPoint,curRoad.end_point,nextPoint)
            curRoad.rho = (rho_start + rho_end) / 2

class Rider:
    def __init__(self, name = "default", gender = True):
        self.name = name
        self.gender = gender
        self.velocity = 10
        self.cur_road = 0 #the index of roads
        self.cur_point = roads[self.cur_road].start_point
        self.cur_dist = 0
        self.total_dist = 0
        self.time = 0
        self.finished = False
    def dist(self,target):
        return np.sum((self.cur_point - target)**2, axis=1, keepdims=True)**0.5
    def update(self, rate = 1):    # 1s
        if self.finished:
            return
        ramain_dis = self.velocity * rate
        while self.cur_dist + ramain_dis > roads[self.cur_road].road_length:
            ramain_dis -= roads[self.cur_road].road_length
            if self.cur_road == len(roads) - 1:
                self.finished = True
            self.cur_road = (self.cur_road + 1) % len(roads)
            self.total_dist += self.cur_dist
            self.cur_dist = 0
            self.cur_point = roads[self.cur_road].start_point
        self.cur_dist += self.velocity * rate
        self.cur_point += self.velocity * rate * roads[self.cur_road].get_dir_vec_norm()
        self.time += 1
        return self.cur_point, self.cur_dist + self.total_dist




def update(i):
    #ax.scatter3D(coods[:, 0],coods[:, 1], coods[:, 2], cmap='Blues')  #绘制散点图
    #plt.clear()
    ax.cla()
    ax.set_xlim([-3000,2000])
    ax.set_ylim([-4000,1000])
    ax.set_zlim([0,1000])
    ax.plot3D(coods[:, 0],coods[:, 1], coods[:, 2], 'gray')
    riders_points = []
    count = 0
    for rider in riders:
        point, dist = rider.update()
        riders_points.append(point)
        time_x.append(rider.time)
        dist_lists[count].append(dist)
        velocity_lists[count].append(rider.velocity)
        ax2.plot(time_x, dist_lists[count], 'r')
        ax3.plot(time_x, velocity_lists[count], 'r')
        ax.text(1000, -3500, 2000 - count * 100, rider.name + ": " +str(datetime.timedelta(seconds=rider.time)))
        count += 1

    if len(riders_points) == 0:
        return
    riders_points = np.array(riders_points)
    ax.scatter3D(riders_points[:, 0], riders_points[:, 1], riders_points[:, 2], 'red')

def get_data():
    f = open("Fuji.csv", "r")
    s = f.read()
    coodStrs = s.split(" ")
    coods = []
    zero_cood = list(map(float,coodStrs[0].split(",")))
    for s1 in coodStrs:
        t = list(map(float,s1.split(",")))
        t[0] = (t[0] - zero_cood[0]) / 0.00001141
        t[1] = (t[1] - zero_cood[1]) / 0.00000899
        coods.append(t)
    f.close()
    return np.array(coods)
        
if __name__ == '__main__':

    coods = get_data()

    global roads
    roads = []

    for i in range(0, len(coods)):
        roads.append(Road(coods[i], coods[(i + 1) % len(coods)]))
    Road.calcRho(roads)
    for i in range(0,len(roads)):
        print(roads[i].project_length,roads[i].rho)

    coods = np.array(coods)
    fig = plt.figure(figsize=(800, 800))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    plt.axis('auto')
    ax.set_xlim([-3000,2000])
    ax.set_ylim([-4000,1000])
    ax.set_zlim([0,1000])

    global riders
    riders = [Rider()]
    dist_lists = []
    velocity_lists = []
    time_x = []
    for i in range(len(riders)):
        dist_lists.append([])
        velocity_lists.append([])

    ax.plot3D(coods[:, 0],coods[:, 1], coods[:, 2], 'gray')
    cur_time = 0
    while cur_time < 10000:
        update(0)
        plt.pause(0.001)
#   # anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=1)
    plt.show()
