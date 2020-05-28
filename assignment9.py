import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def normalize(data):
    data_normalized = data / max(data)

    return (data_normalized)

col_names = ["label"]
for i in range(784):
  col_name = "val " + str(i)
  col_names.append(col_name)

mnist = pd.read_csv("/content/sample_data/mnist.csv", names=col_names)
