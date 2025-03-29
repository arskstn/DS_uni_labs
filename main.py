import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from catboost import CatBoostClassifier
import time
from matplotlib import pyplot as plt

data = pd.read_csv("advanced-dls-spring-2021/train.csv")
test = pd.read_csv("advanced-dls-spring-2021/test.csv")


print(data.head(5))
print(data.info())
print(data.shape)

numerical_fs = ['ClientPeriod', 'MonthlySpending', 'TotalSpent']
categorial_fs = [
    'Sex', 'IsSeniorCitizen', 'HasPartner', 'HasChild', 'HasPhoneService',
    'HasMultiplePhoneNumbers', 'HasInternetService', 'HasOnlineSecurityService',
    'HasOnlineBackup', 'HasDeviceProtection', 'HasTechSupportAccess',
    'HasOnlineTV', 'HasMovieSubscription', 'HasContractPhone',
    'IsBillingPaperless', 'PaymentMethod'
]
feature_columns = numerical_fs + categorial_fs
target = 'Churn'

print(data[data.duplicated(keep=False)])

print(data.isna().sum())
print(data.Churn.value_counts())
plt.pie(data.Churn.value_counts(), autopct='%1.1f%%')
plt.title("Churn Distribution")
# plt.show()

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
for i, j in enumerate(categorial_fs):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    sns.countplot(x=j, hue='Churn', data=data, ax=ax)    
    ax.set_xlabel(j)
    ax.set_ylabel('Количество клиентов')
    ax.set_title(f'Распределение по {j}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
# plt.show()

print(data.dtypes)

print('''
У уходящих нет DeviceProtection, OnlineBackup, детей, партнера
интернет оптоволокно у них, OnlineSecurityService
''')

new_data = data.copy()
binary_cols = [col for col in new_data.columns if set(new_data[col].dropna().unique()) <= {'Yes', 'No'}]
new_data[binary_cols] = new_data[binary_cols].replace({'Yes': 1, 'No': 0})
cat_cols = [col for col in new_data.select_dtypes(include=['object']).columns if new_data[col].nunique() <= 10]
new_data = pd.get_dummies(new_data, columns=cat_cols, drop_first=True)
new_data = new_data.apply(pd.to_numeric, errors='coerce')
new_data = new_data.dropna(axis=1, how='all')
ohe_cols = [col for col in new_data.columns if any(c in col for c in cat_cols)]
num_cols = list(set(new_data.columns) - set(ohe_cols))


plt.figure(figsize=(10, 8))
sns.heatmap(new_data[ohe_cols + ['Churn']].corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8}, linewidths=0.2)
plt.title("Матрица корреляций (OHE + Churn)")
plt.xticks(rotation=90)
# plt.show()


plt.figure(figsize=(10, 8))
corr_matrix = new_data[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_with_churn = corr_matrix['Churn']
sns.heatmap(corr_with_churn.to_frame().sort_values(by='Churn', ascending=False), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 8}, linewidths=0.2)
plt.title("Корреляция Churn с числовыми данными")
plt.xticks(rotation=90)
# plt.show()

# мусорные колонки
data_clean = data.drop(['HasInternetService', 'HasOnlineTV', 'HasMovieSubscription'], axis=1)

for col in numerical_fs:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    if data[col].isna().sum() > 0:
        print(f"Found {data[col].isna().sum()} NaN values in {col}")
        data[col] = data[col].fillna(data[col].median())

# 0/1 и onehot
binary_cols = [col for col in categorial_fs if set(data[col].dropna().unique()) <= {'Yes', 'No'}]
data[binary_cols] = data[binary_cols].replace({'Yes': 1, 'No': 0})
non_binary_cat_cols = [col for col in categorial_fs if col not in binary_cols]
data_encoded = pd.get_dummies(data, columns=non_binary_cat_cols, drop_first=True)

features_train = data_encoded.drop(target, axis=1)
target_train = data_encoded[target]

# убрать дырки надо
if features_train.isna().sum().sum() > 0:
    print("Remaining NaNs in features:", features_train.isna().sum())
    features_train = features_train.dropna()
    target_train = target_train.loc[features_train.index]

X_train, X_test, y_train, y_test = train_test_split(features_train, target_train, test_size=0.2, random_state=42)

# скейлим числовые
scaler = StandardScaler()
X_train[numerical_fs] = scaler.fit_transform(X_train[numerical_fs])
X_test[numerical_fs] = scaler.transform(X_test[numerical_fs])

# логрег
start_time = time.time()
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_time = time.time() - start_time
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
roc_auc_logreg = roc_auc_score(y_test, y_prob_logreg)
print(f"Логистическая регрессия - Время выполнения: {logreg_time:.4f} секунд")
print(f"Логистическая регрессия - ROC AUC: {roc_auc_logreg:.4f}")
print(classification_report(y_test, y_pred_logreg))

# перцептрончег
start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000)
mlp.fit(X_train, y_train)
mlp_time = time.time() - start_time
y_pred_mlp = mlp.predict(X_test)
y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
roc_auc_mlp = roc_auc_score(y_test, y_prob_mlp)
print(f"\nMLP - Время выполнения: {mlp_time:.4f} секунд")
print(f"MLP - ROC AUC: {roc_auc_mlp:.4f}")
print(classification_report(y_test, y_pred_mlp))

# catboost
data_catboost = data.copy()
for col in numerical_fs:
    data_catboost[col] = pd.to_numeric(data_catboost[col], errors='coerce')
    data_catboost[col] = data_catboost[col].fillna(data_catboost[col].median())
cat_features = [data_catboost.columns.get_loc(col) for col in categorial_fs]
X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(
    data_catboost.drop(target, axis=1), data_catboost[target], test_size=0.2, random_state=42
)
start_time = time.time()
catboost = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=0, cat_features=cat_features)
catboost.fit(X_train_cb, y_train_cb)
catboost_time = time.time() - start_time
y_pred_catboost = catboost.predict(X_test_cb)
y_prob_catboost = catboost.predict_proba(X_test_cb)[:, 1]
roc_auc_catboost = roc_auc_score(y_test_cb, y_prob_catboost)
print(f"\nCatBoost - Время выполнения: {catboost_time:.4f} секунд")
print(f"CatBoost - ROC AUC: {roc_auc_catboost:.4f}")
print(classification_report(y_test_cb, y_pred_catboost))

# ROC графики допилить
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
fpr_catboost, tpr_catboost, _ = roc_curve(y_test_cb, y_prob_catboost)
plt.figure(figsize=(10, 8))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_logreg:.4f})')
plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {roc_auc_mlp:.4f})')
plt.plot(fpr_catboost, tpr_catboost, label=f'CatBoost (AUC = {roc_auc_catboost:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()