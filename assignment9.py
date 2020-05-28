import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def normalize(data):
    data_normalized = data / max(data)

    return (data_normalized)

col_names = ["label"]
for i in range(784):
  col_name = "val " + str(i)
  col_names.append(col_name)

mnist = pd.read_csv("/content/sample_data/mnist.csv", names=col_names)

mnist = np.array(mnist)

label_list = mnist[:,0]

target = label_list
num = np.unique(target)
num = num.shape[0]
encoded_label = np.eye(num)[target]
encoded_label = encoded_label.T

training_encoded_label = encoded_label[:,0:6000]
print(len(training_encoded_label[1]))
test_encoded_label = encoded_label[:,4000:]
print(len(test_encoded_label))

training_data = mnist[0:6000]
test_data = mnist[6000:]

# Y = np.random.normal(loc=1, size=(196, 785))
# print(len(training_data[1]))
# y = np.dot(training_data, Y)

# y = sigmoid(y)

# bias = np.ones((1, 6000))

# y = np.vstack([y, bias])

# Z = np.random.normal(loc=1, size=(49, 197))

# z = np.dot(y, Z)

# z = sigmoid(z)

# bias = np.ones((1, 6000))
# z = np.vstack([z, bias])

# P = np.random.normal(loc=1, size=(50, 10))

# p = np.dot(z, P)

# p = sigmoid(p)
# print(p)

train_cost_list = np.empty(0)
train_accuracy_list = np.empty(0)
test_cost_list = np.empty(0)
test_accuracy_list = np.empty(0)

alpha = 0.1

loss = np.empty(0)
cnt = 0

while True:
    if cnt == 10000:
        break

    if cnt % 100 == 0:
        print(cnt)

    bias_x = np.hstack([bias, training_data])
    y = bias_x.dot(u)
    sig_y = sigmoid(y)
    bias_y = np.hstack([bias, sig_y])
    z = bias_y.dot(y)
    sig_z = sigmoid(z)
    bias_z = np.hastack([bias, sig_z])
    h = bias_z.dot(w)
    sig_h = sigmoid(h)
    loss = 0
    accuracy = 0

    for i in range(6000):
        loss += np.mean(-training_encoded_label * np.log(p.T) - (1 - training_encoded_label) * np.log(1 - p.T))
        break




