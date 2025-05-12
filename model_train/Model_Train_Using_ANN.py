import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from django.conf import settings


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # only one level up
train_path = os.path.join(BASE_DIR, 'social_media_train.csv')
test_path = os.path.join(BASE_DIR, 'social_media_test.csv')
model_path = os.path.join(BASE_DIR, 'social_media_model.h5')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
encoders_path = os.path.join(BASE_DIR, 'label_encoders.pkl')

# Paths
# train_path = r'C:\Users\Fatima\Downloads\myproject\main\model_train\social_media_train.csv'
# test_path = r'C:\Users\Fatima\Downloads\myproject\main\model_train\social_media_test.csv'
# model_path = r'C:\Users\Fatima\Downloads\myproject\main\model_train\social_media_model.h5'
# scaler_path = r'C:\Users\Fatima\Downloads\myproject\main\model_train\scaler.pkl'
# encoders_path = r'C:\Users\Fatima\Downloads\myproject\main\model_train\label_encoders.pkl'
# train_path = os.path.join(settings.BASE_DIR, 'main', 'model_train', 'social_media_train.csv')
# test_path = os.path.join(settings.BASE_DIR, 'main', 'model_train', 'social_media_test.csv')
# Load data
instagram_df_train = pd.read_csv(train_path)
instagram_df_test = pd.read_csv(test_path)

# Preprocess
X_train = instagram_df_train.drop(columns=['fake'])
X_test = instagram_df_test.drop(columns=['fake'])
y_train = instagram_df_train['fake']
y_test = instagram_df_test['fake']

label_encoders = {}

# Label Encoding for categorical features
for col in X_train.columns:
    if X_train[col].dtype == 'object' or X_train[col].dtype == 'bool':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le

# Save the label encoders to avoid loading them every time
if not os.path.exists(encoders_path):
    with open(encoders_path, 'wb') as file:
        pickle.dump(label_encoders, file)
else:
    print("Label Encoders already saved, loading existing label encoders...")
    with open(encoders_path, 'rb') as file:
        label_encoders = pickle.load(file)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler to avoid loading it every time
if not os.path.exists(scaler_path):
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
else:
    print("Scaler already saved, loading existing scaler...")
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Check if model already exists
if not os.path.exists(model_path):
    print("Training model...")

    model = Sequential([
        Dense(50, input_dim=X_train.shape[1], activation='relu'),
        Dense(150, activation='relu'),
        Dropout(0.3),
        Dense(150, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)
    model.save(model_path)  # Save the trained model

else:
    print("Loading existing model...")
    model = load_model(model_path)

# Function to predict using preprocessed data
def predict_data():
    # Assuming that we want to predict on the test data (or any new data)
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    # Display the results (Real or Fake)
    results = ['Fake Account 🚨' if predicted_class == 1 else 'Real Account ✅' for predicted_class in predicted_classes]
    
    # Display the first 5 results as an example
    for i in range(50):
        print(f"Profile {i+1}: {results[i]}")

# Perform predictions on the data
predict_data()
