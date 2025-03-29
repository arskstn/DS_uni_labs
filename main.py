import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/advanced-dls-spring-2021/train.csv")
test = pd.read_csv("../input/advanced-dls-spring-2021/test.csv")

#data.sample(5)
data.head(5)
#data.tail(5)
data.info()
data.shape

