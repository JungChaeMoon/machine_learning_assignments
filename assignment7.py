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
