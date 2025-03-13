Bitcoin Price Prediction using Keras Sequential Model

Overview

This project develops a deep learning model using Keras' Sequential API to predict Bitcoin prices. The model analyzes the last 100 days of Bitcoin prices to forecast future values. The dataset is preprocessed, normalized, and used to train an LSTM-based neural network to capture temporal dependencies in the data.

Features

Fetch and preprocess historical Bitcoin price data

Normalize and reshape the data for LSTM model input

Train a deep learning model using Keras Sequential API

Evaluate model performance using relevant metrics

Predict future Bitcoin prices based on past trends

Dataset

The dataset consists of Bitcoin historical price data collected over the last 100 days. It includes features such as:

Date

Open price

High price

Low price

Close price (target variable)

Trading volume

Technologies Used

Python

TensorFlow/Keras

Pandas

NumPy

Scikit-learn

Matplotlib/Seaborn

Installation

Clone the repository:

git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction

Install dependencies:

pip install -r requirements.txt

Model Architecture

The model follows a Sequential architecture using LSTM layers to handle time-series data:

Input layer to accept time-series data

LSTM layers to capture temporal dependencies

Dense layers for prediction

Activation functions such as ReLU and sigmoid

Mean Squared Error (MSE) as the loss function

Adam optimizer for training

Usage

Run the script to train the model:

python train.py

Predict future prices:

python predict.py

Visualize results:

python visualize.py

Results

The trained model provides predictions on future Bitcoin prices with reasonable accuracy. Performance is evaluated using:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Visual comparison with actual prices

Future Improvements

Incorporate more features such as trading volume and sentiment analysis

Experiment with different model architectures (GRU, Transformer, etc.)

Use larger datasets for better generalization

Implement hyperparameter tuning for improved performance

License

This project is licensed under the MIT License.

