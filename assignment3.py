import numpy as np
from matplotlib import pyplot as plt
import csv

data_test_path = "/home/chaemoon/Downloads/data_train.csv"
data_train_path = "/home/chaemoon/Downloads/data_train.csv"

x_data_test = []
y_data_test = []
z_data_test = []
k_data_test = []
x_data_train = []
y_data_train = []
z_data_train = []
k_data_train = []

with open(data_train_path, newline='') as myfile:
    reader = csv.reader(myfile, delimiter=',')
    ct = 1
    for i in reader:
        print('[', ct, ']', 'x =', i[0], ', y = ', i[1], ', z = ', i[2], ', h = ', i[3])
        ct += 1
        x_data_train.append(i[0])
        y_data_train.append(i[1])
        z_data_train.append(i[2])
        k_data_train.append(i[3])

    x_data_train = list(map(float, x_data_train))
    y_data_train = list(map(float, y_data_train))
    z_data_train = list(map(float, z_data_train))
    h_data_train = list(map(float, k_data_train))



with open(data_test_path, newline='') as myfile:
    reader = csv.reader(myfile, delimiter=',')
    ct = 1
    for i in reader:
        print('[', ct, ']', 'x =', i[0], ', y = ', i[1], ', z = ', i[2], ', h = ', i[3])
        ct += 1
        x_data_test.append(i[0])
        y_data_test.append(i[1])
        z_data_test.append(i[2])
        k_data_test.append(i[3])

    x_data_test = list(map(float, x_data_test))
    y_data_test = list(map(float, y_data_test))
    z_data_test = list(map(float, z_data_test))
    h_data_test = list(map(float, k_data_test))

def objective_function(x_data, y_data, z_data, h_data, theta_zero, theta_one, theta_two, theta_three):

    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    theta_two = theta_two
    theta_three = theta_three
    error = 0

    for i in range(0, m):
        error += (theta_zero + (x_data[i] * theta_one) + (y_data[i] * theta_two) + (z_data[i] * theta_three) - h_data[i]) ** 2

    error = error / (2 * m)

    return error

theta_zero = 0
theta_one = 0
theta_two = 1
theta_three = 1
cnt = 0


def gradient_descent_theta_zero(x_data, y_data, z_data, h_data, theta_zero, theta_one, theta_two, theta_three):
    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    theta_two = theta_two
    theta_three = theta_three
    error = 0

    for i in range(0, m):
        error += (theta_zero + (x_data[i] * theta_one) + (y_data[i] * theta_two) + (z_data[i] * theta_three) - h_data[i])

    error = error / m

    return error


def gradient_descent_theta_one(x_data, y_data, z_data, h_data, theta_zero, theta_one, theta_two, theta_three):
    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    theta_two = theta_two
    theta_three = theta_three
    error = 0

    for i in range(0, m):
        error += (theta_zero + (x_data[i] * theta_one) + (y_data[i] * theta_two) + (z_data[i] * theta_three) - h_data[i]) * x_data[i]

    error = error / m

    return error


def gradient_descent_theta_two(x_data, y_data, z_data, h_data, theta_zero, theta_one, theta_two, theta_three):
    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    theta_two = theta_two
    theta_three = theta_three
    error = 0

    for i in range(0, m):
        error += ((theta_zero + (x_data[i] * theta_one) + (y_data[i] * theta_two) + (z_data[i] * theta_three) - h_data[i]) * y_data[i])

    error = error / m

    return error


def gradient_descent_theta_three(x_data, y_data, z_data, h_data, theta_zero, theta_one, theta_two, theta_three):
    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    theta_two = theta_two
    theta_three = theta_three
    error = 0

    for i in range(0, m):
        error += (theta_zero + (x_data[i] * theta_one) + (y_data[i] * theta_two) + (z_data[i] * theta_three) - h_data[i]) * z_data[i]

    error = error / m

    return error


def gradient_function(x_data, y_data, z_data, h_data):
    global theta_one, theta_zero, theta_two, theta_three, cnt
    theta_zero_error = gradient_descent_theta_zero(x_data, y_data,
                                                   z_data, h_data,
                                                   theta_zero, theta_one,
                                                   theta_two, theta_three)
    theta_one_error = gradient_descent_theta_one(x_data, y_data,
                                                 z_data, h_data,
                                                 theta_zero, theta_one,
                                                 theta_two, theta_three)
    theta_two_error = gradient_descent_theta_two(x_data, y_data,
                                                 z_data, h_data,
                                                 theta_zero, theta_one,
                                                 theta_two, theta_three)
    theta_three_error = gradient_descent_theta_three(x_data, y_data,
                                                     z_data, h_data,
                                                     theta_zero, theta_one,
                                                     theta_two, theta_three)
    J = objective_function(x_data, y_data, z_data, h_data, theta_zero, theta_one,
                           theta_two, theta_three)

    check_J = objective_function(x_data_test, y_data_test, z_data_test, h_data_test, theta_zero, theta_one,
                           theta_two, theta_three)
    error_list.append(J)
    check_error_list.append(check_J)
    cur_J = 0

    while True:
        if abs(cur_J - J) < 0.00001:
            break

        J = cur_J
        theta_zero = theta_zero - 0.00002 * theta_zero_error
        theta_one = theta_one - 0.00002 * theta_one_error
        theta_two = theta_two - 0.00002 * theta_two_error
        theta_three = theta_three - 0.00002 * theta_three_error

        theta_zero_error = gradient_descent_theta_zero(x_data, y_data, z_data,
                                                       h_data, theta_zero,
                                                       theta_one, theta_two, theta_three)
        theta_one_error = gradient_descent_theta_one(x_data, y_data, z_data,
                                                     h_data, theta_zero, theta_one,
                                                     theta_two, theta_three)
        theta_two_error = gradient_descent_theta_two(x_data, y_data, z_data,
                                                     h_data, theta_zero,
                                                     theta_one, theta_two,
                                                     theta_three)
        theta_three_error = gradient_descent_theta_three(x_data, y_data, z_data,
                                                         h_data, theta_zero,
                                                         theta_one, theta_two,
                                                         theta_three)
        theta_one_value.append(theta_one)
        theta_zero_value.append(theta_zero)
        theta_two_value.append(theta_two)
        theta_three_value.append(theta_three)
        cur_J = objective_function(x_data, y_data, z_data, h_data,
                                   theta_zero, theta_one,
                                   theta_two, theta_three)
        check_J = objective_function(x_data_test, y_data_test, z_data_test, h_data_test, theta_zero, theta_one,
                                     theta_two, theta_three)
        error_list.append(cur_J)
        check_error_list.append(check_J)
        cnt += 1
        print(cnt)

theta_one_value = []
theta_zero_value = []
theta_two_value = []
theta_three_value = []
error_list = []
check_error_list = []
gradient_function(x_data_train, y_data_train, z_data_train, h_data_train)

