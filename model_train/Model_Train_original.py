# --------- Imports ---------
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use a safe backend (or 'Agg' if running headless)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# --------- Load the Data ---------
# train_path = '/home/Fatima/Downloads/insta_train.csv'
# test_path = '/home/Fatima/Downloads/insta_test.csv'

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # only one level up
train_path = os.path.join(BASE_DIR, 'train.csv')
test_path = os.path.join(BASE_DIR, 'test.csv')


instagram_df_train = pd.read_csv(train_path)
instagram_df_test = pd.read_csv(test_path)

# --------- EDA ---------
print(instagram_df_train.info())
print(instagram_df_train.describe())
print(instagram_df_train.isnull().sum())
print(instagram_df_train['profile pic'].value_counts())
print(instagram_df_train['fake'].value_counts())

# --------- Data Visualization ---------
sns.countplot(x='fake', data=instagram_df_train)
plt.title("Fake vs Real Accounts")
plt.show()

sns.countplot(x='private', data=instagram_df_train)
plt.title("Private Accounts Distribution")
plt.show()

sns.countplot(x='profile pic', data=instagram_df_train)
plt.title("Profile Picture Presence")
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(instagram_df_train['nums/length username'], kde=True)
plt.title("Username Length Distribution")
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(instagram_df_train.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# --------- Preprocessing ---------

# Drop target column
X_train = instagram_df_train.drop(columns=['fake'])
X_test = instagram_df_test.drop(columns=['fake'])
y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']

# Encode boolean/categorical columns if needed
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode output labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# --------- Model Definition ---------
model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------- Model Training ---------
history = model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)

# --------- Plot Training Progress ---------
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --------- Predictions and Evaluation ---------
predicted = model.predict(X_test)
predicted_labels = np.argmax(predicted, axis=1)
true_labels = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(true_labels, predicted_labels))

# --------- Confusion Matrix ---------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()