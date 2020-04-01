import numpy as np
from matplotlib import pyplot as plt


x_data = np.random.normal(loc=0.0, scale=2.0, size=(100, 1)) #input_data  평균 0 표준편차2 랜덤 x_data 생성
y_hat = [(x * 2) for x in x_data] # a = 2, b = 0 선정한후 y_hat 데이터 생성 type(np,darray)
y_data = [y + np.random.normal(loc=0.0, scale=2.0) for y in y_hat] #평균 0 표준편차 2인 난수생성으로 y_hat 속성값에 더하기
plt.scatter(x=x_data, y=y_data, edgecolors='black') #분산된 x_data, y_data 더하기


def draw_model_parameters(theta_zero_value, theta_one_value, cnt):
    plt.plot([i for i in range(0, cnt)], theta_zero_value, color='blue')
    plt.plot([i for i in range(0, cnt)], theta_one_value, color='red')
    plt.show()


def objective_function(x_data, y_data, theta_one, theta_zero):

    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    error = 0

    for i in range(0, m):
        error += ((x_data[i][0] * theta_one + theta_zero) - y_data[i]) ** 2

    error = error / (2 * m)

    return error


def finely_divided_function(x_data, y_data, theta_one, theta_zero):

    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    error = 0

    for i in range(0, m):
        error += ((x_data[i][0] * theta_one + theta_zero) - y_data[i])

    error = error / m

    return error


def finely_divided_function_one(x_data, y_data, theta_one, theta_zero):
    m = 100
    theta_zero = theta_zero
    theta_one = theta_one
    error = 0

    for i in range(0, m):
        error += (((x_data[i][0] * theta_one + theta_zero) - y_data[i]) * x_data[i][0])

    error = error / m

    return error


def gradient_function(x_data, y_data):

    theta_zero = 4
    theta_one = 5
    theta_zero_error = finely_divided_function(x_data, y_data, theta_one, theta_zero)
    theta_one_error = finely_divided_function_one(x_data, y_data,theta_one, theta_zero)
    J = objective_function(x_data, y_data, theta_one=theta_one, theta_zero=theta_zero)
    error_list.append(J)
    cur_J = 0
    cnt = 0

    while True:
        if cur_J == J:
          break

        J = cur_J
        theta_zero = theta_zero - 0.0005 * theta_zero_error
        theta_one = theta_one - 0.0005 * theta_one_error
        theta_zero_error = finely_divided_function(x_data, y_data, theta_one, theta_zero)
        theta_one_error = finely_divided_function_one(x_data, y_data, theta_one, theta_zero)
        theta_one_value.append(theta_one)
        theta_zero_value.append(theta_zero)
        cur_J = objective_function(x_data, y_data, theta_one, theta_zero)
        error_list.append(cur_J)
        cnt += 1

    y_value_list = [(x[0] * theta_one + theta_zero)for x in x_data]
    plt.plot(x_data, y_value_list, color='red')
    plt.scatter(x=x_data, y=y_data, color='blue')
    plt.show()
    draw_energy_value(error_list, cnt)


x_data = np.random.normal(loc=0.0, scale=8.0, size=(100, 1)) #input_data  평균 0 표준편차2 랜덤 x_data 생성
y_hat = [(x[0] * 2) for x in x_data]# a = 2, b = 0 선정한후 y_hat 데이터 생성 type(np,darray)
y_data = [y + np.random.normal(loc=0.0, scale=8.0) for y in y_hat] #평균 0 표준편차 2인 난수생성으로 y_hat 속성값에 더하기
plt.scatter(x=x_data, y=y_data, color='black') #분산된 x_data, y_data 점찍기
plt.plot(x_data, y_hat, color='blue')# y_hat = 2x linear graph 그리기
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
theta_one_value = []
theta_zero_value = []
error_list = []
cnt = 0
gradient_function(x_data, y_data)
