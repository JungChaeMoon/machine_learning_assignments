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