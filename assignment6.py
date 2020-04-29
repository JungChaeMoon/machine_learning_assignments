import numpy as np
import matplotlib.pyplot as plt
from math import e, log

data    = np.genfromtxt("/content/sample_data/data-nonlinear.txt", delimiter=',')

pointX  = data[:, 0]
pointY  = data[:, 1]
label   = data[:, 2]

pointX0 = pointX[label == 0]
pointY0 = pointY[label == 0]

pointX1 = pointX[label == 1]
pointY1 = pointY[label == 1]

plt.figure()
plt.scatter(pointX0, pointY0, c='b')
plt.scatter(pointX1, pointY1, c='r')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

m = len(pointX)
alpha = 0.01
cnt = 0
theta_zero = -10
theta_one = 1
theta_two = 1
theta_three = 1
theta_four = 1
theta_five = 1
theta_six = 1
theta_seven = 1
theta_eight = 1
theta_nine = 1
theta_ten = 1
theta_ele = 1
theta_twe = 1
sigmoid_value_list = []
error_list_value = []
theta_zero_list = [0, ]
theta_one_list = [0, ]
theta_two_list = [0, ]
theta_three_list = [0, ]
theta_four_list = [0, ]
theta_five_list = [0, ]
theta_six_list = [0, ]
theta_seven_list = [0, ]
theta_eight_list = [0, ]
theta_nine_list = [0, ]
theta_ten_list = [0, ]

flag = False
# x^{2\ }+\ y^{2\ }+x^{2}y^{2}+x^{6}y^{2}\ +\ x^{6}y^{4}\ +\ y^{6}\ +\ x^{2}y^{4}\ +\ x^{4\ }+\ x^{8}\ +xy^{4}\-10=\ 0
# x^{2\ }+\ y^{2\ }+x^{2}y^{2}+x^{4}y^{4}\ +\ x^{2}y^{4}\ +\ y^{6}\ +\ x^{6}y^{8}=\ 10

def sigmoid(i):
    global pointX, pointY
    temp = 1 / (1 + np.exp(-(theta_zero + theta_one * pointX[i] ** 2 + theta_two * pointY[i] ** 2 + theta_three * (pointX[i] ** 2) * (pointY[i] ** 2) + theta_four * (pointX[i] ** 6) * (pointY[i] ** 2) + theta_five * (pointX[i] ** 6) * (pointY[i] ** 4) + theta_six * pointY[i] ** 6 + theta_seven * pointX[i] ** 2 * pointY[i] ** 4 + theta_eight * pointX[i] ** 4 + theta_nine * pointX[i] ** 8 + theta_ten * pointX[i] * pointY[i] ** 4)))

    sigmoid_value_list.append(temp)
    return temp


def objective_function():
    global pointX, label, m
    error = 0

    for i in range(0, m):
        error += (-label[i] * log(sigmoid(i))) - (1 - label[i]) * log(1 - sigmoid(i))

    error = error / m
    error_list_value.append(error)
    print(error)
    return error


def gradient_descent_theta_zero():
    global theta_zero, m, theta_zero_list
    sum_value = 0
    for i in range(m):
        sum_value += sigmoid(i) - label[i]

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_one():
    global theta_one, m, theta_one_list, pointX
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * pointX[i] ** 2

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_two():
    global theta_two, m, theta_two_list, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * pointY[i] ** 2

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_three():
    global theta_three, m, theta_three_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i] ** 2) * (pointY[i] ** 2)

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_four():
    global theta_four, m, theta_four_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i] ** 6) * (pointY[i] ** 2)

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_five():
    global theta_five, m, theta_five_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i] ** 6) * (pointY[i] ** 4)

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_six():
    global theta_six, m, theta_six_list, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointY[i] ** 6)

    sum_value = sum_value / m
    return sum_value


def gradient_descent_theta_seven():
    global theta_seven, m, theta_seven_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i] ** 2) * (pointY[i] ** 4)

    sum_value = sum_value / m
    return sum_value

def gradient_descent_theta_eight():
    global theta_eight, m, theta_eight_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i] ** 4)

    sum_value = sum_value / m
    return sum_value

def gradient_descent_theta_nine():
    global theta_nine, m, theta_nine_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i] ** 8)

    sum_value = sum_value / m
    return sum_value

def gradient_descent_theta_ten():
    global theta_ten, m, theta_ten_list, pointX, pointY
    sum_value = 0
    for i in range(m):
        sum_value += (sigmoid(i) - label[i]) * (pointX[i]) * (pointY[i] ** 4)

    sum_value = sum_value / m
    return sum_value


def gradient_descent():
    global theta_zero, theta_one, theta_two, theta_three, theta_four, theta_five, theta_six, theta_seven, cnt

    theta_zero_error = gradient_descent_theta_zero()
    theta_one_error = gradient_descent_theta_one()
    theta_two_error = gradient_descent_theta_two()
    theta_three_error = gradient_descent_theta_three()
    theta_four_error = gradient_descent_theta_four()
    theta_five_error = gradient_descent_theta_five()
    theta_six_error = gradient_descent_theta_six()
    theta_seven_error = gradient_descent_theta_seven()
    theta_eight_error = gradient_descent_theta_eight()
    theta_nine_error = gradient_descent_theta_nine()
    theta_ten_error = gradient_descent_theta_ten()

    J = objective_function()
    cur_J = 0

    while True:
        # if abs(J - cur_J) < 0.000001:
        if cnt > 1000000:
            break
        J = cur_J
        theta_zero = theta_zero - alpha * theta_zero_error
        theta_one = theta_one - alpha * theta_one_error
        theta_two = theta_two - alpha * theta_two_error
        theta_three = theta_three - alpha * theta_three_error
        theta_four = theta_four - alpha * theta_four_error
        theta_five = theta_five - alpha * theta_five_error
        theta_six = theta_six - alpha * theta_six_error
        theta_seven = theta_seven - alpha * theta_seven_error
        theta_eight = theta_seven - alpha * theta_eight_error
        theta_nine = theta_seven - alpha * theta_nine_error
        theta_ten = theta_seven - alpha * theta_ten_error

        theta_zero_error = gradient_descent_theta_zero()
        theta_one_error = gradient_descent_theta_one()
        theta_two_error = gradient_descent_theta_two()
        theta_three_error = gradient_descent_theta_three()
        theta_four_error = gradient_descent_theta_four()
        theta_five_error = gradient_descent_theta_five()
        theta_six_error = gradient_descent_theta_six()
        theta_seven_error = gradient_descent_theta_seven()
        theta_eight_error = gradient_descent_theta_eight()
        theta_nine_error = gradient_descent_theta_nine()
        theta_ten_error = gradient_descent_theta_ten()

        theta_zero_list.append(theta_zero)
        theta_one_list.append(theta_one)
        theta_two_list.append(theta_two)
        theta_three_list.append(theta_three)
        theta_four_list.append(theta_four)
        theta_five_list.append(theta_five)
        theta_six_list.append(theta_six)
        theta_seven_list.append(theta_seven)
        theta_eight_list.append(theta_eight)
        theta_nine_list.append(theta_nine)
        theta_ten_list.append(theta_ten)


        cur_J = objective_function()
        cnt += 1
        print(cnt)


gradient_descent()
