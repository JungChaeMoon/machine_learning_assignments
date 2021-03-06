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


def sigmoid(x):
  return 1 / (1 + np.exp(-x))
X = np.random.normal(loc=0, scale=1, size=len(avg_img[0]))

total_val = []
for i in range(len(digit_list)):
  val = 0
  for row in digit_list[i]:
    val += sigmoid(row.dot(X))
  total_val.append(val/len(digit_list[i]))

for i in range(len(avg_img)):
  plt.subplot(2,5,i+1)
  plt.imshow(np.array(avg_img[i]).reshape(28,28), cmap="Greys")
plt.show()

col_names = []
for i in range(10):
  col_name = "label" + str(i)
  col_names.append(col_name)

tv_df = pd.DataFrame(total_val, columns=['value'],index=col_names)

