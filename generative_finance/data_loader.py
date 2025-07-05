import yfinance as yf
import numpy as np
import pandas as pd
# My choice to linearly scale was primarily driven by the apparent sensitivity
# of GANs to the scale of input data. Scaling this way aids in stabilizing
# the training process and faster convergence.
# Link -> https://ydata.ai/resources/synthetic-time-series-data-a-gan-approach
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(ticker: str, start_date: str, end_date: str, seq_len: int):
    """
    Loads, preprocesses, and shapes financial time-series data for generative models.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        seq_len (int): The length of the input sequences for the model.

    Returns:
        tuple: A tuple containing:
            - processed_data (np.ndarray): The normalized and sequenced data.
            - scaler (MinMaxScaler): The scaler object used for normalization.
    """
    # 1. Gather data from yfinance
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Please check symbol or data ranges.")

    # 2. Select the 'Close' price and handle the missing values
    data = df[['Close']].dropna()

    # 3. Normalization to the [0, 1] scale
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    # 4. Create overlapping sequences for TS modeling
    sequences = []
    for i in range(len(scaled_data) - seq_len + 1):
        sequences += [scaled_data[i:i+seq_len]]

    # Convert list to a np array for input
    processed_data = np.array(sequences)

    print(f"Data process into {processed_data.shape[0]} sequences of length {seq_len}.")

    return processed_data, scaler