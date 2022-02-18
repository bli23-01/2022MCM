import code
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

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
print(coods)

coods = np.array(coods)
fig = plt.figure()
ax1 = plt.axes(projection='3d')

plt.axis('auto')
ax1.set_xlim([-3000,2000])
ax1.set_ylim([-4000,1000])
ax1.set_zlim([0,1000])

ax1.scatter3D(coods[:, 0],coods[:, 1], coods[:, 2], cmap='Blues')  #绘制散点图
ax1.plot3D(coods[:, 0],coods[:, 1], coods[:, 2], 'gray') 
plt.show()

f.close()

class road:
    def __init__(self, start_point, end_point):
        self.start_point = start_point #标准化之后的点信息
        self.end_point = end_point
        self.height = end_point[2] - start_point[2] 
        self.project_length = math.sqrt((start_point[0] - end_point[0]) ** 2 + 
                                        (start_point[1] - end_point[1]) ** 2)
        self.road_length = math.sqrt(self.project_length ** 2 + 
                                    self.height ** 2)
        self.theta = math.atan(self.height / self.project_length)

class rider:
    def __init__(self, gender):
        self.gender = gender

