import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


class Assignment1():

    theta_zero = 4
    theta_one = 5
    theta_one_value = []
    theta_zero_value = []
    error_list = []
    Z = []
    flag = False

    def objective_function(self, x_data, y_data, theta_one, theta_zero):
        m = 100
        theta_zero = theta_zero
        theta_one = theta_one
        error = 0

        for i in range(0, m):
            error += ((x_data[i] * theta_one + theta_zero) - y_data[i]) ** 2

        error = error / (2 * m)

        if self.flag:
            self.Z.append(error)

        return error

    def gradient_descent_theta_zero(self, x_data, y_data, theta_one, theta_zero):

        m = 100
        theta_zero = theta_zero
        theta_one = theta_one
        error = 0

        for i in range(0, m):
            error += ((x_data[i] * theta_one + theta_zero) - y_data[i])

        error = error / m

        return error

    def gradient_descent_theta_one(self, x_data, y_data, theta_one, theta_zero):
        m = 100
        theta_zero = theta_zero
        theta_one = theta_one
        error = 0

        for i in range(0, m):
            error += (((x_data[i] * theta_one + theta_zero) - y_data[i]) * x_data[i])

        error = error / m

        return error

    def gradient_function(self, x_data, y_data):
        theta_zero_error = self.gradient_descent_theta_zero(x_data, y_data, self.theta_one, self.theta_zero)
        theta_one_error = self.gradient_descent_theta_one(x_data, y_data, self.theta_one, self.theta_zero)
        J = self.objective_function(x_data, y_data, theta_one=self.theta_one, theta_zero=self.theta_zero)
        self.error_list.append(J)
        cur_J = 0
        cnt = 0

        while True:
            if cur_J == J:
                break

            J = cur_J
            self.theta_zero = self.theta_zero - 0.01 * theta_zero_error
            self.theta_one = self.theta_one - 0.01 * theta_one_error
            theta_zero_error = self.gradient_descent_theta_zero(x_data, y_data, self.theta_one, self.theta_zero)
            theta_one_error = self.gradient_descent_theta_one(x_data, y_data, self.theta_one, self.theta_zero)
            self.theta_one_value.append(self.theta_one)
            self.theta_zero_value.append(self.theta_zero)
            cur_J = self.objective_function(x_data, y_data, self.theta_one, self.theta_zero)
            self.error_list.append(cur_J)
            cnt += 1

    def run(self):
        path = "data.csv"
        data = np.genfromtxt(path, delimiter=',')

        x_data = data[:, 0]
        y_data = data[:, 1]

        plt.figure(figsize=(8, 8))
        plt.scatter(x_data, y_data, color='black')
        plt.show()
        self.gradient_function(x_data, y_data)
        y_value_list = [(x * self.theta_one + self.theta_zero) for x in x_data]
        plt.plot(x_data, y_value_list, color='red')
        plt.scatter(x=x_data, y=y_data, color='black')
        plt.show()

        self.flag = True
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(-30, 30, 0.1)
        Y = np.arange(-30, 30, 0.1)
        self.objective_function(x_data, y_data, X, Y)
        X, Y = np.meshgrid(X, Y)
        self.Z = np.array(self.Z)
        surf = ax.plot_surface(X, Y, self.Z, cmap='coolwarm', linewidth=0, antialiased=False)
        wire = ax.plot_wireframe(X, Y, self.Z, color='r', linewidth=0.1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        fig.tight_layout()
        plt.show()
        np.arange()


if __name__ == '__main__':
    Assignment1().run()
