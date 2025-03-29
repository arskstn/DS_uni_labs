import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("advanced-dls-spring-2021/train.csv")
test = pd.read_csv("advanced-dls-spring-2021/test.csv")

#data.sample(5)
data.head(5)
#data.tail(5)
data.info()
data.shape

numerical_fs = [
    'ClientPeriod',
    'MonthlySpending',
    'TotalSpent'
]

categorial_fs = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_columns = numerical_fs + categorial_fs
target = 'Churn'
data[data.duplicated(keep=False)]