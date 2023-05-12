import os
import ccxt
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk

# Constants
exchange = ccxt.binance()
lookback_hours = 94
limit = 1000  # candles
model_directory = 'models/'
model_name_template = '{}_{}_model.h5'
scaler_name_template = '{}_{}_scaler.pkl'
interval_time = 90000  # in ms
symbol_options = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT',
                  'ADA/USDT', 'XRP/USDT', 'MATIC/USDT', 'AGIX/USDT', 'BNB/USDT']
time_frame_options = ['5m', '15m', '1h', '4h', '1d']

# Default symbol and time frame
symbol = 'BTC/USDT'
time_frame = '1h'


def get_scaler(symbol, time_frame):
    scaler_name = scaler_name_template.format(
        symbol, time_frame).replace('/', '_')
    scaler_path = os.path.join(model_directory, scaler_name)

    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        historical_data = exchange.fetch_ohlcv(symbol, time_frame, limit=limit)
        prices = np.array([row[4] for row in historical_data])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(prices.reshape(-1, 1))
        joblib.dump(scaler, scaler_path)

    return scaler


def get_model(symbol, time_frame):
    model_name = model_name_template.format(
        symbol, time_frame).replace('/', '_')
    model_path = os.path.join(model_directory, model_name)

    if os.path.isfile(model_path):
        return tf.keras.models.load_model(model_path)

    scaler = get_scaler(symbol, time_frame)
    historical_data = exchange.fetch_ohlcv(symbol, time_frame, limit=limit)
    prices = np.array([row[4] for row in historical_data])
    scaled_prices = scaler.transform(prices.reshape(-1, 1))
    return train(scaled_prices, model_path)


def get_data(symbol, time_frame):
    scaler = get_scaler(symbol, time_frame)
    model = get_model(symbol, time_frame)
    historical_data = exchange.fetch_ohlcv(symbol, time_frame, limit=limit)
    prices = np.array([row[4] for row in historical_data])
    scaled_prices = scaler.transform(prices.reshape(-1, 1))
    return scaler, model, scaled_prices


def train(scaled_prices, model_path):
    X_train = []
    y_train = []
    for i in range(lookback_hours, len(scaled_prices)):
        X_train.append(scaled_prices[i - lookback_hours:i, 0])
        y_train.append(scaled_prices[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=200, activation='relu',
                              input_shape=(lookback_hours,)),
        tf.keras.layers.Dense(units=200, activation='relu'),
        tf.keras.layers.Dense(units=4),
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, batch_size=128)
    model.save(model_path)
    return model


def update_accuracy():
    global scheduled_accuracy_update_id
    X_test = []
    y_test = []
    for i in range(lookback_hours, len(scaled_prices)):
        X_test.append(scaled_prices[i - lookback_hours:i, 0])
        y_test.append(scaled_prices[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    loss = model.evaluate(X_test, y_test)
    accuracy = 1 - loss
    root.after(0, lambda: accuracy_label.configure(
        text="Model Accuracy: {:.2%}".format(accuracy)))
    scheduled_accuracy_update_id = root.after(interval_time, update_accuracy)


def update_prediction():
    global prices, scheduled_prediction_update_id
    recent_prices = scaler.transform(prices[-limit:].reshape(-1, 1))
    X_predict = recent_prices[-lookback_hours:].reshape(1, -1)
    next_price = scaler.inverse_transform(model.predict(X_predict))[0][0]
    actual_price = exchange.fetch_ohlcv(symbol, time_frame, limit=2)[-1][4]
    if next_price > prices[-1]:
        root.after(0, lambda: predicted_price_label.configure(
            text=f"The predicted price for the next {time_frame} is: ${next_price:,.2f} \u2191"))
    else:
        root.after(0, lambda: predicted_price_label.configure(
            text=f"The predicted price for the next {time_frame} is: ${next_price:,.2f} \u2193"))
    prices = np.append(prices, actual_price)

    scheduled_prediction_update_id = root.after(
        interval_time, update_prediction)


scheduled_accuracy_update_id = None
scheduled_prediction_update_id = None


def update_data(new_symbol, new_time_frame):
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
    prices = np.array(
        [row[4] for row in exchange.fetch_ohlcv(symbol, time_frame, limit=limit)])
    predicted_price_label.configure(
        text=f"The predicted price for the next {time_frame} is: LOADING...")
    accuracy_label.configure(text="Model Accuracy: LOADING...")

    # Schedule the initial execution of the update_accuracy() and update_prediction() functions
    scheduled_accuracy_update_id = root.after(0, update_accuracy)
    scheduled_prediction_update_id = root.after(0, update_prediction)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


create_directory('models')

root = tk.Tk()
root.title("Crypto Price Predictor")

scaler, model, scaled_prices = get_data(symbol, time_frame)
prices = np.array(
    [row[4] for row in exchange.fetch_ohlcv(symbol, time_frame, limit=limit)])

symbol_var = tk.StringVar(value=symbol)
symbol_menu = tk.OptionMenu(root, symbol_var, *symbol_options,
                            command=lambda symbol: update_data(symbol, time_frame))
symbol_menu.pack()

time_frame_var = tk.StringVar(value=time_frame)
time_frame_menu = tk.OptionMenu(root, time_frame_var, *time_frame_options,
                                command=lambda time_frame: update_data(symbol, time_frame))
time_frame_menu.pack()

predicted_price_label = tk.Label(
    root, text=f"The predicted price for the next {time_frame} is: LOADING...", font=("Helvetica", 16))
predicted_price_label.pack(padx=20, pady=5)

accuracy_label = tk.Label(
    root, text="Model Accuracy: LOADING...", font=("Helvetica", 14))
accuracy_label.pack(padx=20, pady=5)

root.after(0, update_accuracy)
root.after(0, update_prediction)

root.mainloop()
