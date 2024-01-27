# app.py
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Assuming 'startup_forest_model.pkl' is in the same directory
# Load your trained model
model = pd.read_pickle('startup_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['csv_file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Read CSV file
    df = pd.read_csv(file)

    # Perform prediction for the entire dataset
    total_profit = model.predict(df).sum()  # Adjust this based on your model's input requirements

    return render_template('index.html', total_profit=total_profit)

if __name__ == '__main__':
    app.run(debug=True)
