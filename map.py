import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

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
    def get_dir_vec_norm(self):
        v = self.end_point - self.start_point
        return v / np.sqrt(np.sum(v**2))

class Rider:
    def __init__(self, gender = True):
        self.gender = gender
        self.velocity = 10
        self.cur_road = 0 #the index of roads
        self.cur_point = roads[self.cur_road].start_point
        self.cur_dist = 0
    def dist(self,target):
        return np.sum((self.cur_point - target)**2, axis=1, keepdims=True)**0.5
    def update(self, rate = 1):    # 1s
        ramain_dis = self.velocity * rate
        while self.cur_dist + ramain_dis > roads[self.cur_road].road_length:
            ramain_dis -= roads[self.cur_road].road_length
            self.cur_road = (self.cur_road + 1) % len(roads)
            self.cur_dist = 0
            self.cur_point = roads[self.cur_road].start_point
        self.cur_dist += self.velocity * rate
        self.cur_point += self.velocity * rate * roads[self.cur_road].get_dir_vec_norm()
        return self.cur_point
        
def update(i):
    #ax.scatter3D(coods[:, 0],coods[:, 1], coods[:, 2], cmap='Blues')  #绘制散点图
    #plt.clear()
    plt.cla()
    ax.set_xlim([-3000,2000])
    ax.set_ylim([-4000,1000])
    ax.set_zlim([0,1000])
    ax.plot3D(coods[:, 0],coods[:, 1], coods[:, 2], 'gray')
    riders_points = []
    for rider in riders:
        riders_points.append(rider.update())
    riders_points = np.array(riders_points)
    ax.scatter3D(riders_points[:, 0], riders_points[:, 1], riders_points[:, 2], 'red')
    surf = ax.plot_surface(curve[:, 0], curve[:, 1], curve[:, 2], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

def get_data():
    f = open("data.csv", "r")
    s = f.read()
    coodStrs = s.split(" ")
    coods = []
    curve = []
    zero_cood = list(map(float,coodStrs[0].split(",")))
    for s1 in coodStrs:
        t = list(map(float,s1.split(",")))
        t[0] = (t[0] - zero_cood[0]) / 0.00001141
        t[1] = (t[1] - zero_cood[1]) / 0.00000899
        coods.append(t)
        curve.append(t)
        curve.append([t[0], t[1], 0])
    f.close()
    return np.array(coods), np.array(curve)
        
if __name__ == '__main__':

    coods, curve = get_data()
    print(np.shape(curve))

    global roads
    roads = []
    

    for i in range(0, len(coods)):
        roads.append(Road(coods[i], coods[(i + 1) % len(coods)]))
        #print(roads[i].road_length)

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
    cur_time = 0
    surf = ax.plot_surface(curve[:, 0], curve[:, 1], curve[:, 2], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #while cur_time < 10000:
    #    update(0)
    #    plt.pause(0.001)
#    anim = FuncAnimation(fig, update, frames=np.arange(0, 10), interval=1)
    plt.show()
