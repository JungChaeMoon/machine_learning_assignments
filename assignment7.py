import numpy as np
import sys
import matplotlib.pyplot as plt
from math import e, log

data = np.genfromtxt("/content/sample_data/data-nonlinear.txt", delimiter=',')

pointX = data[:, 0]
pointY = data[:, 1]
label = data[:, 2]

pointX0 = pointX[label == 0]
pointY0 = pointY[label == 0]

pointX1 = pointX[label == 1]
pointY1 = pointY[label == 1]

data_xy_val = []

for t in range(len(pointX)):
  tmp = []
  for i in range(10):
    for j in range(10):
      tmp.append(pointX[t] ** i * pointY[t] ** j)
  data_xy_val.append(tmp)

m = len(data_xy_val)
alpha = 0.01
underfit_lambda = 1

cnt = 0

underfit_theta_list = np.zeros(100)
underfit_J_list = []
underfit_accuracy_list = []
overfit_lambda = 0.0001
overfit_theta_list = np.zeros(100)
overfit_J_list = []
overfit_accuracy_list = []
just_write_lambda = 0.001
just_write_theta_list= np.zeros(100)
just_write_J_list = []
just_write_accuracy_list = []

while True:
    if cnt % 10000 == 0:
        print(cnt)

    if cnt > 30000:
        break
    ## compute sum val per one data!
    underfit_z_sigmoid = []

    for i in range(len(data_xy_val)):
        underfit_z = 0
        for j in range(len(data_xy_val[i])):
            underfit_z += underfit_theta_list[j] * data_xy_val[i][j]
        underfit_z_sigmoid.append(1 / (1 + np.exp(-(underfit_z))))

    h_val = np.array(underfit_z_sigmoid)

    underfit_J = np.sum(-label * np.log(h_val) - (1 - label) * np.log(1 - h_val)) / m + np.sum(
        underfit_theta_list ** 2 * underfit_lambda / 2)
    underfit_J_list.append(underfit_J)

    correct = 0
    for i in range(m):
        if underfit_z_sigmoid[i] < 0.5:
            if label[i] == 0:
                correct += 1
        elif underfit_z_sigmoid[i] >= 0.5:
            if label[i] == 1:
                correct += 1
    underfit_accuracy_list.append(correct / m * 100)

    for i in range(10):
        for j in range(10):
            underfit_theta_list[10 * i + j] = (1 - alpha * underfit_lambda) * underfit_theta_list[
                10 * i + j] - alpha * (np.sum((underfit_z_sigmoid - label) * pointX ** i * pointY ** j) / m)
    cnt += 1

cnt = 0
while True:
    if cnt % 1000 == 0:
        print(cnt)

    if cnt > 30000:
        break
    ## compute sum val per one data!
    overfit_z_sigmoid = []

    for i in range(len(data_xy_val)):
        overfit_z = 0
        for j in range(len(data_xy_val[i])):
            overfit_z += overfit_theta_list[j] * data_xy_val[i][j]
        overfit_z_sigmoid.append(1 / (1 + np.exp(-(overfit_z))))

    h_val = np.array(overfit_z_sigmoid)
    overfit_J = np.sum(-label * np.log(h_val) - (1 - label) * np.log(1 - h_val)) / m + np.sum(
        overfit_theta_list ** 2 * overfit_lambda / 2)
    overfit_J_list.append(overfit_J)

    correct = 0
    for i in range(m):
        if overfit_z_sigmoid[i] < 0.5:
            if label[i] == 0:
                correct += 1
        elif overfit_z_sigmoid[i] >= 0.5:
            if label[i] == 1:
                correct += 1
    overfit_accuracy_list.append(correct / m * 100)

    for i in range(10):
        for j in range(10):
            overfit_theta_list[10 * i + j] = (1 - alpha * overfit_lambda) * overfit_theta_list[10 * i + j] - alpha * (
                        np.sum((overfit_z_sigmoid - label) * pointX ** i * pointY ** j) / m)
    cnt += 1

cnt = 0
while True:
    if cnt % 1000 == 0:
        print(cnt)

    if cnt > 30000:
        break
    ## compute sum val per one data!
    just_write_z_sigmoid = []

    for i in range(len(data_xy_val)):
        just_write_z = 0
        for j in range(len(data_xy_val[i])):
            just_write_z += just_write_theta_list[j] * data_xy_val[i][j]
        just_write_z_sigmoid.append(1 / (1 + np.exp(-(just_write_z))))

    h_val = np.array(just_write_z_sigmoid)
    just_write_J = np.sum(-label * np.log(h_val) - (1 - label) * np.log(1 - h_val)) / m + np.sum(
        just_write_theta_list ** 2 * just_write_lambda / 2)
    just_write_J_list.append(just_write_J)

    correct = 0
    for i in range(m):
        if just_write_z_sigmoid[i] < 0.5:
            if label[i] == 0:
                correct += 1
        elif just_write_z_sigmoid[i] >= 0.5:
            if label[i] == 1:
                correct += 1
    just_write_accuracy_list.append(correct / m * 100)

    for i in range(10):
        for j in range(10):
            just_write_theta_list[10 * i + j] = (1 - alpha * just_write_lambda) * just_write_theta_list[
                10 * i + j] - alpha * (np.sum((just_write_z_sigmoid - label) * pointX ** i * pointY ** j) / m)
    cnt += 1

plt.plot([i for i in range(0, len(overfit_J_list))], overfit_J_list, color='red')
plt.plot([i for i in range(0, len(just_write_J_list))], just_write_J_list, color='green')
plt.plot([i for i in range(0, len(underfit_J_list))], underfit_J_list, color='blue')

plt.show()

RED   = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE  = "\033[1;34m"

sys.stdout.write(RED)
print("over-fitting lambda {}".format(overfit_lambda))
sys.stdout.write(GREEN)
print("just-right-fitting lambda {}".format(just_write_lambda))
sys.stdout.write(BLUE)
print("under-fitting lambda {}".format(underfit_lambda))


plt.plot([i for i in range(len(overfit_accuracy_list))], overfit_accuracy_list, color='red')
plt.plot([i for i in range(len(just_write_accuracy_list))], just_write_accuracy_list, color='green')
plt.plot([i for i in range(len(underfit_accuracy_list))], underfit_accuracy_list, color='blue')
plt.show()

RED   = "\033[1;31m"
GREEN = "\033[0;32m"
BLUE  = "\033[1;34m"

sys.stdout.write(RED)
print(" final accuracy is {}, over-fitting lambda is {}%".format(overfit_accuracy_list[-1], overfit_lambda))
sys.stdout.write(GREEN)
print(" final accuracy is {}, just-write-fitting lambda is {}%".format(just_write_accuracy_list[-1], just_write_lambda))
sys.stdout.write(BLUE)
print(" final accuracy is {}, under-fitting lambda is {}%".format(underfit_accuracy_list[-1], underfit_lambda))



X = np.arange(-1, 1, 0.001)
Y = np.arange(-1, 1, 0.001)
xx, yy = np.meshgrid(X, Y)

underfitting_Z = 0.0
overfitting_Z = 0.0
just_write_fitting_Z = 0.0

for i in range(10):
  for j in range(10):
    underfitting_Z += underfit_theta_list[i*10 + j] * xx ** i * yy ** j
    overfitting_Z += overfit_theta_list[i*10+j] * xx **i * yy ** j
    just_write_fitting_Z += just_write_theta_list[i*10+j] * xx **i * yy ** j


plt.contour(xx,yy,underfitting_Z, [0], colors='green' )
plt.scatter(pointX0, pointY0, c='b')
plt.scatter(pointX1, pointY1, c='r')
plt.show()

plt.contour(xx,yy,overfitting_Z, [0], colors='blue' )
plt.scatter(pointX0, pointY0, c='b')
plt.scatter(pointX1, pointY1, c='r')
plt.show()


plt.contour(xx,yy,just_write_fitting_Z, [0], colors='red' )
plt.scatter(pointX0, pointY0, c='b')
plt.scatter(pointX1, pointY1, c='r')
plt.show()
