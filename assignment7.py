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

