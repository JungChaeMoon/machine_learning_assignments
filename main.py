import numpy as np
from matplotlib import pyplot as plt


x_data = np.random.normal(loc=0.0, scale=2.0, size=(100, 1)) #input_data  평균 0 표준편차2 랜덤 x_data 생성
y_hat = [(x * 2) for x in x_data] # a = 2, b = 0 선정한후 y_hat 데이터 생성 type(np,darray)
y_data = [y + np.random.normal(loc=0.0, scale=2.0) for y in y_hat] #평균 0 표준편차 2인 난수생성으로 y_hat 속성값에 더하기
plt.scatter(x=x_data, y=y_data, edgecolors='black') #분산된 x_data, y_data 더하기
plt.plot(x_data, y_hat, color='blue') # y_hat = 2x linear graph 그리기
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
