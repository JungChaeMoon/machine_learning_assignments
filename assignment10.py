import matplotlib.pyplot as plt
import numpy as np

file_data   = "/content/sample_data/mnist.csv"
handle_file = open(file_data, "r")
data        = handle_file.readlines()
handle_file.close()

size_row    = 28    # height of the image
size_col    = 28    # width of the image

num_image   = len(data)
count       = 0     # count for the number of images

#
# normalize the values of the input data to be [0, 1]
#
def normalize(data):

    data_normalized = (data - min(data)) / (max(data) - min(data))

    return(data_normalized)

#
# example of distance function between two vectors x and y
#
def distance(x, y):

    d = (x - y) ** 2
    s = np.sum(d)
    # r = np.sqrt(s)

    return(s)

#
# make a matrix each column of which represents an images in a vector form
#
list_image  = np.empty((size_row * size_col, num_image), dtype=float)
list_label  = np.empty(num_image, dtype=int)

for line in data:

    line_data   = line.split(',')
    label       = line_data[0]
    im_vector   = np.asfarray(line_data[1:])
    im_vector   = normalize(im_vector)

    list_label[count]       = label
    list_image[:, count]    = im_vector

    count += 1

one_hot_encoding = [
  [1,0,0,0,0,0,0,0,0,0],
  [0,1,0,0,0,0,0,0,0,0],
  [0,0,1,0,0,0,0,0,0,0],
  [0,0,0,1,0,0,0,0,0,0],
  [0,0,0,0,1,0,0,0,0,0],
  [0,0,0,0,0,1,0,0,0,0],
  [0,0,0,0,0,0,1,0,0,0],
  [0,0,0,0,0,0,0,1,0,0],
  [0,0,0,0,0,0,0,0,1,0],
  [0,0,0,0,0,0,0,0,0,1],
]

one_hot_encoding = np.array(one_hot_encoding)
train_image = np.empty((size_row * size_col, 1000), dtype=float)
test_image = np.empty((size_row * size_col, 9000), dtype=float)

for i in range(num_image):
  if i>= 1000:
    test_image[:, i-1000] = list_image[:, i]
  else:
    train_image[:, i] = list_image[:, i]

training_label = []
testing_label = []
for j in range(num_image):
  label_value = list_label[j]
  if j<1000:
    training_label.append(one_hot_encoding[label_value])
  else:
    testing_label.append(one_hot_encoding[label_value])

training_label = np.array(training_label).T
testing_label = np.array(testing_label).T

def sigmoid(x):
  return 1 / (1+np.exp(-x))

def sigmoid_p(x):
  return sigmoid(z) * (1 - sigmoid(z))

u = np.random.normal(0,1, 196*784).reshape((196,784))
v = np.random.normal(0,1, 49*196).reshape((49,196))
w = np.random.normal(0,1, 10*49).reshape((10,49))

b1 = np.zeros((196,1))
b2 = np.zeros((49,1))
b3 = np.zeros((10,1))
# u = np.random.normal(0,1, 196*784).reshape((196,784))
# v = np.random.normal(0,1, 49*196).reshape((49,196))
# w = np.random.normal(0,1, 10*49).reshape((10,49))

# b1 = np.zeros((196,1))
# b2 = np.zeros((49,1))
# b3 = np.zeros((10,1))

# u_bias = np.ones((1000, 1))
# v_bias = np.ones((1000, 1))
# w_bias = np.ones((1000, 1))

# test_x_bias = np.ones((9000, 1))
# test_u_bias = np.ones((9000, 1))
# test_v_bias = np.ones((9000, 1))
# test_w_bias = np.ones((9000, 1))

# new_u_bias = 0.0
# new_v_bias = 0.0
# new_w_bias = 0.0
def accuracy(label, sig):
  correct = 0
  length = len(label[0])
  for i in range(length):
    if label[sig[:, i].argmax(), i] == 1:
      correct += 1
  return (correct/length) * 100


a = 0.4
lambda_value = 0.5

n = 784 * 196 + 196 * 49 + 49 * 10

J_list = []
test_J_list = []
accuracy_list = []
test_accuracy_list = []
count = 0
flag = True

while True:
    # training step is 3000
    if count == 3000:
        break

    J = 0
    test_J = 0

    y_ = np.dot(u, train_image) + b1
    y = sigmoid(y_)
    z_ = np.dot(v, y) + b2
    z = sigmoid(z_)
    h_ = np.dot(w, z) + b3
    h = sigmoid(h_)
    # calculate loss about train, test data
    J = (1 / 1000) * np.sum(-((training_label) * np.log(h) + (1 - training_label) * np.log(1 - h))) + lambda_value * (
                1 / (2 * n)) * (np.sum(u * u) + np.sum(v * v) + np.sum(w * w))
    J_list.append(J)

    test_y_ = np.dot(u, test_image) + b1
    test_y = sigmoid(test_y_)
    test_z_ = np.dot(v, test_y) + b2
    test_z = sigmoid(test_z_)
    test_h_ = np.dot(w, test_z) + b3
    test_h = sigmoid(test_h_)
    test_J = (1 / 9000) * np.sum(
        -((testing_label) * np.log(test_h) + (1 - testing_label) * np.log(1 - test_h))) + lambda_value * (
                         1 / (2 * n)) * (np.sum(u * u) + np.sum(v * v) + np.sum(w * w))
    test_J_list.append(test_J)

    # calculate gradient descent and update

    df_h_ = h - training_label
    df_w = (1 / 1000) * (np.dot(df_h_, z.T) + (lambda_value) * w)
    w -= a * df_w

    df_z_ = np.multiply(np.dot(w.T, df_h_), z * (1 - z))
    df_v = (1 / 1000) * (np.dot(df_z_, y.T) + (lambda_value) * v)
    v -= a * df_v

    df_y_ = np.multiply(np.dot(v.T, df_z_), y * (1 - y))
    df_u = (1 / 1000) * (np.dot(df_y_, train_image.T) + (lambda_value) * u)
    u -= a * df_u

    df_b3 = (1 / 1000) * np.sum(df_h_, axis=1, keepdims=True)
    df_b2 = (1 / 1000) * np.sum(df_z_, axis=1, keepdims=True)
    df_b1 = (1 / 1000) * np.sum(df_y_, axis=1, keepdims=True)

    b3 -= a * df_b3
    b2 -= a * df_b2
    b1 -= a * df_b1

    # calculate the accuracy about train, test data
    accuracy_list.append(accuracy(training_label, h))
    test_accuracy_list.append(accuracy(testing_label, test_h))
    print(accuracy(testing_label, test_h))

    count += 1