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
