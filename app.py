import os
import ccxt
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk

# Configuration
config = {
    "exchange": ccxt.binance(),
    "lookback_hours": 94,
    "limit": 1000,  # candles
    "model_directory": 'models/',
    "model_name_template": '{}_{}_model.h5',
    "scaler_name_template": '{}_{}_scaler.pkl',
    "interval_time": 90000,  # in ms
    "symbol_options": ['BTC/USDT', 'ETH/USDT', 'LTC/USDT', 'ADA/USDT', 'XRP/USDT', 'MATIC/USDT', 'AGIX/USDT', 'BNB/USDT', 'DOT/USDT'],
    "time_frame_options": ['5m', '15m', '1h', '4h', '1d', '1w'],
    "default_symbol": 'XRP/USDT',
    "default_time_frame": '4h'
}

scheduled_accuracy_update_id = None
scheduled_prediction_update_id = None
root = None
accuracy_label = None
predicted_price_label = None
scaler = None
model = None
scaled_prices = None
prices = None


def create_directory(directory_path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_scaler(symbol, time_frame):
    """Loads or creates a MinMaxScaler for the given symbol and time frame."""
    scaler_name = config["scaler_name_template"].format(
        symbol, time_frame).replace('/', '_')
    scaler_path = os.path.join(config["model_directory"], scaler_name)

    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        historical_data = config["exchange"].fetch_ohlcv(
            symbol, time_frame, limit=config["limit"])
        prices = np.array([row[4] for row in historical_data])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(prices.reshape(-1, 1))
        joblib.dump(scaler, scaler_path)

    return scaler


def get_model(symbol, time_frame):
    """Loads or creates a model for the given symbol and time frame."""
    model_name = config["model_name_template"].format(
        symbol, time_frame).replace('/', '_')
    model_path = os.path.join(config["model_directory"], model_name)

    if os.path.isfile(model_path):
        return tf.keras.models.load_model(model_path)

    scaler = get_scaler(symbol, time_frame)
    historical_data = config["exchange"].fetch_ohlcv(
        symbol, time_frame, limit=config["limit"])
    prices = np.array([row[4] for row in historical_data])
    scaled_prices = scaler.transform(prices.reshape(-1, 1))
    return train(scaled_prices, model_path)


def get_data(symbol, time_frame):
    """Gets the scaler, model, and scaled prices for the given symbol and time frame."""
    scaler = get_scaler(symbol, time_frame)
    model = get_model(symbol, time_frame)
    historical_data = config["exchange"].fetch_ohlcv(
        symbol, time_frame, limit=config["limit"])
    prices = np.array([row[4] for row in historical_data])
    scaled_prices = scaler.transform(prices.reshape(-1, 1))
    return scaler, model, scaled_prices


def train(scaled_prices, model_path):
    """Trains a model using the given scaled prices and saves it to the given path."""
    X_train = []
    y_train = []
    for i in range(config["lookback_hours"], len(scaled_prices)):
        X_train.append(scaled_prices[i - config["lookback_hours"]:i, 0])
        y_train.append(scaled_prices[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=200, activation='relu',
                              input_shape=(config["lookback_hours"],)),
        tf.keras.layers.Dense(units=200, activation='relu'),
        tf.keras.layers.Dense(units=4),
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, batch_size=128)
    model.save(model_path)
    return model


def update_accuracy():
    """Updates the accuracy of the model."""
    global scheduled_accuracy_update_id
    X_test = []
    y_test = []
    for i in range(config["lookback_hours"], len(scaled_prices)):
        X_test.append(scaled_prices[i - config["lookback_hours"]:i, 0])
        y_test.append(scaled_prices[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    loss = model.evaluate(X_test, y_test)
    accuracy = 1 - loss
    root.after(0, lambda: accuracy_label.configure(
        text="Model Accuracy: {:.2%}".format(accuracy)))
    scheduled_accuracy_update_id = root.after(
        config["interval_time"], update_accuracy)


def update_prediction():
    """Updates the prediction of the model."""
    global prices, scheduled_prediction_update_id
    recent_prices = scaler.transform(prices[-config["limit"]:].reshape(-1, 1))
    X_predict = recent_prices[-config["lookback_hours"]:].reshape(1, -1)
    next_price = scaler.inverse_transform(model.predict(X_predict))[0][0]
    actual_price = config["exchange"].fetch_ohlcv(
        symbol, time_frame, limit=2)[-1][4]
    if next_price > prices[-1]:
        root.after(0, lambda: predicted_price_label.configure(
            text="The predicted price for the next {} is: ${:,.5f} \u2191".format(time_frame, next_price)))
    else:
        root.after(0, lambda: predicted_price_label.configure(
            text="The predicted price for the next {} is: ${:,.5f} \u2193".format(time_frame, next_price)))
    prices = np.append(prices, actual_price)

    scheduled_prediction_update_id = root.after(
        config["interval_time"], update_prediction)


def update_data(new_symbol, new_time_frame):
    """Updates the data for the given symbol and time frame."""
    global symbol, time_frame, scaler, model, scaled_prices, prices
    global scheduled_accuracy_update_id, scheduled_prediction_update_id

    # Cancel any pending update_accuracy() and update_prediction() calls
    if scheduled_accuracy_update_id is not None:
        root.after_cancel(scheduled_accuracy_update_id)
    if scheduled_prediction_update_id is not None:
        root.after_cancel(scheduled_prediction_update_id)

    symbol = new_symbol
    time_frame = new_time_frame
    scaler, model, scaled_prices = get_data(symbol, time_frame)
    prices = np.array([row[4] for row in config["exchange"].fetch_ohlcv(
        symbol, time_frame, limit=config["limit"])])
    predicted_price_label.configure(
        text="The predicted price for the next {} is: LOADING...".format(time_frame))
    accuracy_label.configure(text="Model Accuracy: LOADING...")

    # Schedule the initial execution of the update_accuracy() and update_prediction() functions
    scheduled_accuracy_update_id = root.after(0, update_accuracy)
    scheduled_prediction_update_id = root.after(0, update_prediction)


def create_gui():
    """Creates the GUI for the application."""
    global root, accuracy_label, predicted_price_label
    root = tk.Tk()
    root.title("Crypto Price Predictor")

    symbol_var = tk.StringVar(value=config["default_symbol"])
    symbol_menu = tk.OptionMenu(
        root, symbol_var, *config["symbol_options"], command=lambda symbol: update_data(symbol, config["default_time_frame"]))
    symbol_menu.pack()

    time_frame_var = tk.StringVar(value=config["default_time_frame"])
    time_frame_menu = tk.OptionMenu(
        root, time_frame_var, *config["time_frame_options"], command=lambda time_frame: update_data(config["default_symbol"], time_frame))
    time_frame_menu.pack()

    predicted_price_label = tk.Label(root, text="The predicted price for the next {} is: LOADING...".format(
        config["default_time_frame"]), font=("Helvetica", 16))
    predicted_price_label.pack(padx=20, pady=5)

    accuracy_label = tk.Label(
        root, text="Model Accuracy: LOADING...", font=("Helvetica", 14))
    accuracy_label.pack(padx=20, pady=5)

    update_data(config["default_symbol"], config["default_time_frame"])

    root.mainloop()


create_directory(config["model_directory"])
create_gui()
