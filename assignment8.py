import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

col_names = ["label"]
for i in range(784):
  col_name = "val " + str(i)
  col_names.append(col_name)

mnist = pd.read_csv("mnist_test.csv", names = col_names)

