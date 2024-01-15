from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

app = Flask(__name__)

# Load the trained model and scaler
model = tf.keras.models.load_model('my_model')
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        x_1 = float(request.form['size'])
        x_2 = int(request.form['Bedrooms'])
        x_new = np.array([[x_1, x_2]])
        x_new_scaled = scaler.transform(x_new)
        prediction = model.predict(x_new_scaled)[0][0]

        # Handle potential NaN predictions
        if np.isnan(prediction):
            prediction = "Error: Invalid input or model issue."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
