# !pip install --upgrade tensorflow jax jaxlib # run to make sure there are no errors
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


model = load_model("best_model.keras")

sample_data = {
    'id': 128411,
    'year-month': '2008-12',
    'total_consumption': 1994,
    'avg_monthly_consumption': 1994.0,
    'consumption_std': 0.0,
    'consumption_change_rate': 0.0,
    'months_since_last_invoice': 0.0,
    'monthly_invoice_count': 1,
    'target': 1
}

# Note that the above data still has to undergo reshaping for LSTM input
# This is done in the function below
def predict(client):
    client = pd.DataFrame(client, index = [0])


    # Start of data prep for LSTM
    feature_cols = [
    "total_consumption", "avg_monthly_consumption", "consumption_std",
    "consumption_change_rate", "months_since_last_invoice", "monthly_invoice_count"
    ]

    client_sequences = {}
    client_labels = {}

    for client_id, group in client.groupby("id"):
        seq = group[feature_cols].values.tolist()
        client_sequences[client_id] = seq
        client_labels[client_id] = group["target"].iloc[0]

    # Finding max seq length for padding
    max_seq_length = 30

    # Porting dict values to lists
    X_sequences = []
    y_labels = []

    for client_id in client_sequences:
        X_sequences.append(client_sequences[client_id])
        y_labels.append(client_labels[client_id])

    # Padding all shorter sequences using the max length
    X_padded = pad_sequences(X_sequences, maxlen = max_seq_length, dtype = 'float32', padding = 'post', truncating = 'post')
    X_padded = np.array(X_padded)

    y = np.array(y_labels)
    num_features = X_padded.shape[2] # No. of features

    unique, counts = np.unique(y, return_counts = True)
    print(dict(zip(unique, counts)))

    n_samples, max_seq_length, num_features = X_padded.shape

    # Flatten
    X_padded_flat = X_padded.reshape(-1, num_features)

    scaler = StandardScaler()
    X_padded_flat_scaled = scaler.fit_transform(X_padded_flat)

    # Reshape
    X_padded_scaled = X_padded_flat_scaled.reshape(n_samples, max_seq_length, num_features)

    # End of data prep


    # Predictions using LSTM model
    prediction = model.predict(X_padded_scaled)
    prediction_output = "client is fraudulent" if prediction[0][0] > 0.5 else "client is legitimate"
    return prediction_output

predict(sample_data)
