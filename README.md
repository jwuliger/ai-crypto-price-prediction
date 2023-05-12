# Crypto Price Predictor

Crypto Price Predictor is a Python application which uses machine learning (a basic deep learning model) to predict the future price of various cryptocurrencies. The tool fetches historical price data using the CCXT library to interact with the Binance API, trains a model on this data using TensorFlow, and then uses this model to predict future prices.

## Features

1. Fetches historical data from Binance.
2. Trains a simple deep learning model on the historical data.
3. Predicts future prices based on the trained model.
4. Provides a basic GUI to select the cryptocurrency and the time frame for the prediction.

## Requirements

1. Python 3.7 or higher
2. Libraries: CCXT, numpy, joblib, tensorflow, sklearn, tkinter

You can install these libraries using pip:

```bash
pip install ccxt numpy joblib tensorflow scikit-learn tkinter
```
## How to Use

1. Clone the repository or download the script.
2. Open a terminal and navigate to the directory containing the script.
3. Run the script using Python:

```bash
python app.py
```

4. The application will open in a new window. Select the cryptocurrency and time frame of your choice from the drop-down menus. The application will display the predicted price and the model's accuracy.

## Disclaimer

This is a simple deep learning model intended for educational purposes. It should not be used for making real-world investment decisions. Always do your own research before investing in cryptocurrencies or any other asset.

## Contributions

Contributions to improve this project are welcome. Please feel free to fork the repository and submit a pull request with your changes.