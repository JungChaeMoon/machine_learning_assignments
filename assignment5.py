import numpy as np
from matplotlib import pyplot as plt
from math import e, log

data    = np.genfromtxt("data.txt", delimiter=',')

x = data[:, 0]
y = data[:, 1]
label = data[:, 2]

x_label0 = x[label == 0]
x_label1 = x[label == 1]

y_label0 = y[label == 0]
y_label1 = y[label == 1]

plt.figure(figsize=(8, 8))
plt.scatter(x_label0, y_label0, alpha=0.3, c='b')
plt.scatter(x_label1, y_label1, alpha=0.3, c='r')
plt.show()

m = len(x)
alpha = 0.005
cnt = 0
theta_zero = 0
theta_one = 0
theta_two = 0
sigmoid_value_list = []
error_list_value = []
theta_zero_list = [0, ]
theta_one_list = [0, ]
theta_two_list = [0, ]


def sigmoid(i):
    global x, y
    temp = 1 / (1 + e ** -(theta_zero + theta_one * x[i] + theta_two * y[i]))
    sigmoid_value_list.append(temp)
    return temp


def objective_function():
    global x, label, m
    error = 0

    for i in range(0, m):
        error += (-label[i] * log(sigmoid(i))) - (1-label[i]) * log(1-sigmoid(i))

    error = error / m
    error_list_value.append(error)
    return error


def gradient_descent_theta_zero():
    global theta_zero, m, theta_zero_list
    sum_value = 0
    for i in range(m):
        sum_value += sigmoid(i) - label[i]

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_one():
    global theta_one, m, theta_one_list, x
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * x[i]

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_two():
    global theta_two, m, theta_two_list, y
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * y[i]

    sum_value = sum_value / m
    return sum_value


def gradient_descent():
    global theta_zero, theta_one, theta_two, cnt

    theta_zero_error = gradient_descent_theta_zero()
    theta_one_error = gradient_descent_theta_one()
    theta_two_error = gradient_descent_theta_two()

    J = objective_function()
    cur_J = 0

    while True:
        if J == cur_J:
            break
        J = cur_J
        theta_zero = theta_zero - alpha * theta_zero_error
        theta_one = theta_one - alpha * theta_one_error
        theta_two = theta_two - alpha * theta_two_error

        theta_zero_error = gradient_descent_theta_zero()
        theta_one_error = gradient_descent_theta_one()
        theta_two_error = gradient_descent_theta_two()

        theta_zero_list.append(theta_zero)
        theta_one_list.append(theta_one)
        theta_two_list.append(theta_two)

        cur_J = objective_function()
        cnt += 1

gradient_descent()
plt.plot([i for i in range(0, cnt+1)], theta_zero_list, color='black')
plt.plot([i for i in range(0, cnt+1)], theta_one_list, color='red')
plt.plot([i for i in range(0, cnt+1)], theta_two_list, color='green')
plt.show()

plt.plot([i for i in range(0, cnt+1)], error_list_value, color='blue')
plt.show()

X = np.arange(30, 100, 0.5)
Y = np.arange(30, 100, 0.5)
sigma_list = []
length = len(X)

def sigma():
  global theta_one, theta_zero, theta_two
  for x in X:
    sigma_list = []
    for y in Y:
      temp = 1 / (1 + e ** -(theta_zero + theta_one * x + theta_two * y))
      sigma_list.append(temp)

    plt.scatter([x]*140, Y, c=mad, alpha=0.06 ,cmap='bwr')

sigma()

plt.scatter(x_label0, y_label0 ,c= 'b')
plt.scatter(x_label1, y_label1, c= 'r')
plt.show()
