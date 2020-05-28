import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# file_data = "/content/sample_data/mnist.csv"
# handle_file = open(file_data, "r")
# data = handle_file.readlines()
# handle_file.close()

# size_row = 28 # height of the image
# size_col = 28 # width of the image

# num_image = len(data)
# count = 0 # count for the number of images

# #
# # normalize the values of the input data to be [0, 1]
# #
# def normalize(data):

# data_normalized = (data - min(data)) / (max(data) - min(data))

# return(data_normalized)

def normalize(data):

  data_normalized = data/max(data)

  return (data_normalized)

# #
# # example of distance function between two vectors x and y
# #
# def distance(x, y):

# d = (x - y) ** 2
# s = np.sum(d)
# # r = np.sqrt(s)

# return(s)

# #
# # make a matrix each column of which represents an images in a vector form
# #
# list_image = np.empty((size_row * size_col, num_image),
col_names = ["label"]
for i in range(784):
  col_name = "val " + str(i)
  col_names.append(col_name)
mnist = pd.read_csv("/content/sample_data/mnist.csv", names = col_names)
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
print(training_encoded_label.shape)
training_data = mnist[:6000]
test_data = mnist[6000:]
# training_data_list = []

# for i in range(10):
# td = training_data[training_data[:,0] == i]
# td = np.array(td)
# training_data_list.append(td[:,1:])

# test_data_list = []

# for i in range(10):
# td = test_data[test_data[:,0] == i]
# td = np.array(td)
# test_data_list.append(td[:,1:])

# test_data_list
train_image = np.empty((6000, 784), dtype=float)
training_data_label = training_data[:,0]
training_data = training_data[:,1:]
for i in range(len(training_data)):
  training_data[i, :] = normalize(training_data[i,:])
  print(training_data.shape)
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def loss(label, h):
  return np.mean(-label*np.log(h) - (1-label)*np.log(1-h))

bias = np.ones((6000,1))

training_data = np.c_[training_data, bias]
Y = np.random.normal(0, 1, 785 * 196).resize((196, 785))
Z = np.random.normal(0, 1, 197 * 49).resize((49, 197))
P = np.random.normal(0, 1, 50 * 10).resize((10, 50))

learning = 1.25

while True:
  y = np.dot(Y,training_data.T)
  y = sigmoid(y)

  z = np.dot(Z,y)
  z = sigmoid(z)

  p = np.dot(P,z)
  p = sigmoid(p)

  l = -np.mean(training_encoded_label*np.log(p) + (1-training_encoded_label)*np.log(1-p))
  print(l)

