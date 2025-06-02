import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Загрузка меток
labels = pd.read_csv('archive/labels_fdd_test.csv')
print(labels.head())

# Создание словаря для быстрого доступа к файлам данных
data_dict = {}
for idx, row in labels.iterrows():
    file_name = row['id']
    data_path = f'archive/data_test/{file_name}'
    data_dict[file_name] = pd.read_csv(data_path)

# Нормализация функции (пример)
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min())

# Определение последовательности и нормализация данных
sequence_length = 10  # Выберите подходящую длину последовательности

X, y = [], []

for idx, row in labels.iterrows():
    file_name = row['id']
    label = row['category']
    
    df = data_dict[file_name]
    df.drop(columns=['H2'], inplace=True)  # Удаляем столбец H2
    normalized_df = normalize_data(df)
    
    if len(normalized_df) >= sequence_length:
        for i in range(len(normalized_df) - sequence_length + 1):
            X.append(normalized_df.iloc[i:i+sequence_length].values)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f'X shape: {X.shape}, y shape: {y.shape}')

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train shapes - X: {X_train.shape}, y: {y_train.shape}')
print(f'Test shapes - X: {X_test.shape}, y: {y_test.shape}')

# Создание и компиляция модели LSTM
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    Dropout(0.5),
    LSTM(units=32, return_sequences=False),
    Dropout(0.5),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')