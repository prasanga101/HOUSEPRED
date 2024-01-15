import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("data.csv")

# Preprocess the data
x_train = df.drop('House_Price', axis=1)
y_train = df['House_Price']

# Initialize and fit the scaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=25, activation='linear', input_shape=(len(x_train.columns),)),
    tf.keras.layers.Dense(units=15, activation='linear'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

model.summary()

# Compile and train the model
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['mae', 'mse'])

history = model.fit(x_train_scaled, y_train, epochs=100, validation_split=0.1)

# Save the trained model
model.save("my_model")

# Save the scaler mean and scale
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
x_1 = float(input("Enter the size of the house in feet (100): "))
x_2 = int(input("Enter the number of bedrooms in the house: "))
x_new = np.array([[x_1, x_2]])

# Load the saved model
model = tf.keras.models.load_model("my_model")

# Load the saved scaler mean and scale
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')

# Use the loaded mean and scale to set up the scaler
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

x_new_scaled = scaler.transform(x_new)

# Make predictions using the loaded model and scaler
predictions = model.predict(x_new_scaled)

print("Predicted Price of the House is (in 1000):")
print(predictions)
