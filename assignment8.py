import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

col_names = ["label"]
for i in range(784):
  col_name = "val " + str(i)
  col_names.append(col_name)

mnist = pd.read_csv("mnist_test.csv", names = col_names)

mnist = np.array(mnist)
digit_list = []
for i in range(10):
  digit = mnist[mnist[:,0] == i]
  digit = np.array(digit)
  digit_list.append(digit[:,1:])

avg_img = []
for digit_info in digit_list:
  avg_digit = []
  for i in range(len(digit_info.T)):
    val = np.sum(digit_info[:,i])
    avg_digit.append(val/len(digit_info))
  avg_img.append(avg_digit)

