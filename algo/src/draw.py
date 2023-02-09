import matplotlib.pyplot as plt
import numpy as np

plt.xlim(1,100)
plt.ylim(0,0.2)
#绘制图形
x1 = []
y1 = []
k=1
for i in np.loadtxt('../output/GD.txt'):
    y1.append(i)
    x1.append(k)
    k=k+1
x2 = []
y2 = []
k=1
for i in np.loadtxt('../output/srNesterov.txt'):
    y2.append(i)
    x2.append(k)
    k=k+1
x3 = []
y3 = []
k=1
for i in np.loadtxt('../output/ftrsNesterov.txt'):
    y3.append(i)
    x3.append(k)
    k=k+1
x4 = []
y4 = []
k=1
for i in np.loadtxt('../output/Nesterov.txt'):
    y4.append(i)
    x4.append(k)
    k=k+1
x5 = []
y5 = []
k=1
for i in np.loadtxt('../output/rNesterov.txt'):
    y5.append(i)
    x5.append(k)
    k=k+1
x6 = []
y6 = []
k=1
for i in np.loadtxt('../output/fsNesterov.txt'):
    y6.append(i)
    x6.append(k)
    k=k+1
# x2,y2 = readandret("model&data\LeNet_0\los.txt")
# # x3,y3 = readandret("model&data\LeNet_2\los.txt")
# # x4,y4 = readandret("outputs\out_tanh.txt")
plt.plot(x1,y1, c='b',label = "GradientDescent")
plt.plot(x2,y2, c='r',label = "SpeedRestartNesterov")
plt.plot(x3,y3, c='g',label = "RestartatRegularIntervalsNesterov")
plt.plot(x4,y4, c=[0,0,0],label = "Nesterov")
plt.plot(x5,y5, c=[1,1,0],label = "GeneralizedNesterov,r=5")
plt.plot(x6,y6, c=[1,0,1],label = "GradientRestartNesterov")
# # plt.plot(x4,y4,c=[0.5,0.5,0.5],label = "tanh")
plt.xlabel("iters")
plt.ylabel("loss")
plt.legend()
plt.show()