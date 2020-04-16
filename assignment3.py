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

