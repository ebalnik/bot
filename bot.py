import ccxt
import time
import logging
import pandas as pd
import requests
import threading
from flask import Flask, render_template
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import numpy as np
from transformers import TFGPT2Model, GPT2Tokenizer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor  # Замена RandomForestRegressor на XGBoost
import openai

# Настройка логирования
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Загрузка конфигурации
config = {
    'apiKey': 'H5KtK7G1GYI15q4FAD',  # Замените на ваш API Key
    'secret': 'JUd5BVIuIKLd2sAv7wI58nQZIPZcqN2ArZ63',  # Замените на ваш Secret Key
    'risk_per_trade': 0.02,  # Риск на сделку (2% от баланса)
    'rsi_period': 14,  # Период для RSI
    'check_interval': 60,  # Интервал проверки (в секундах)
    'telegram_token': '7821156218:AAERY-obfoYsd41fFaQXWmNjx5L4isUIfuE',  # Токен Telegram бота
    'telegram_chat_id': '500482633',  # ID чата в Telegram
    'openai_api_key': 'ваш_OpenAI_API_Key',  # API ключ OpenAI
    'max_pairs': 5,  # Максимальное количество пар для торговли
    'max_daily_loss': 0.05,  # Максимальный дневной убыток (5%)
}

# Настройка подключения к Bybit
exchange = ccxt.bybit({
    'apiKey': config['apiKey'],
    'secret': config['secret'],
})

# Веб-интерфейс
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def send_telegram_message(message):
    """Отправка сообщения в Telegram."""
    url = f"https://api.telegram.org/bot{config['telegram_token']}/sendMessage"
    params = {
        'chat_id': config['telegram_chat_id'],
        'text': message,
    }
    try:
        requests.post(url, params=params)
    except Exception as e:
        logging.error(f"Error sending Telegram message: {e}")

def get_historical_data(symbol, timeframe='1h', limit=100):
    """Получение исторических данных."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Расчет RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Расчет MACD."""
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd.iloc[-1], signal.iloc[-1]

def calculate_bollinger_bands(df, window=20, num_std=2):
    """Расчет Bollinger Bands."""
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band.iloc[-1], lower_band.iloc[-1]

def calculate_atr(df, period=14):
    """Расчет Average True Range (ATR)."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr.iloc[-1]

def predict_price_gpt4(df):
    """Прогнозирование цены с использованием GPT-4."""
    try:
        openai.api_key = config['openai_api_key']
        prompt = f"Predict the next price for {df['symbol'].iloc[-1]} based on the following data:\n{df.tail(10)}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50
        )
        predicted_price = float(response.choices[0].message['content'].strip())
        return predicted_price
    except Exception as e:
        logging.error(f"Error predicting price with GPT-4: {e}")
        return None

def optimize_hyperparameters(df):
    """Оптимизация гиперпараметров с использованием XGBoost."""
    try:
        X = df[['open', 'high', 'low', 'close', 'volume']]
        y = df['close']
        model = XGBRegressor()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
        grid_search = GridSearchCV(model, param_grid, cv=3)
        grid_search.fit(X, y)
        return grid_search.best_params_
    except Exception as e:
        logging.error(f"Error optimizing hyperparameters: {e}")
        return None

def auto_adjust_parameters(df):
    """Автоматическая настройка параметров."""
    # Настройка RSI
    rsi = calculate_rsi(df['close'], config['rsi_period'])
    rsi_overbought = rsi + 10  # Уровень перекупленности
    rsi_oversold = rsi - 10  # Уровень перепроданности
    # Настройка стоп-лосса и тейк-профита на основе ATR
    atr = calculate_atr(df)
    stop_loss = atr * 1.5  # Стоп-лосс: 1.5 * ATR
    target_profit = atr * 2  # Тейк-профит: 2 * ATR
    return {
        'rsi_overbought': rsi_overbought,
        'rsi_oversold': rsi_oversold,
        'stop_loss': stop_loss,
        'target_profit': target_profit,
    }

def get_balance():
    """Получение баланса."""
    try:
        balance = exchange.fetch_balance()
        return balance['total']['USDT']
    except Exception as e:
        logging.error(f"Error fetching balance: {e}")
        return None

def calculate_position_size(balance, risk_per_trade, stop_loss):
    """Расчет размера позиции."""
    risk_amount = balance * risk_per_trade
    position_size = risk_amount / stop_loss
    return position_size

def place_order_with_stop_loss(symbol, order_type, amount, stop_loss_price):
    """Размещение ордера с лимитным стоп-лоссом."""
    try:
        order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side=order_type,
            amount=amount,
            price=stop_loss_price,
            params={'stopLoss': stop_loss_price}
        )
        logging.info(f"Order placed with stop-loss: {order}")
        send_telegram_message(f"Order placed with stop-loss: {order}")
        return order
    except Exception as e:
        logging.error(f"Error placing order with stop-loss: {e}")
        send_telegram_message(f"Error placing order with stop-loss: {e}")
        return None

def select_trading_pairs():
    """Выбор торговых пар на основе волатильности и ликвидности."""
    try:
        markets = exchange.load_markets()
        pairs = []
        for symbol in markets:
            if markets[symbol]['active'] and markets[symbol]['quote'] == 'USDT':
                df = get_historical_data(symbol)
                if df is not None:
                    atr = calculate_atr(df)
                    volume = df['volume'].mean()
                    pairs.append({
                        'symbol': symbol,
                        'atr': atr,
                        'volume': volume,
                    })
        # Сортировка пар по волатильности и ликвидности
        pairs.sort(key=lambda x: x['atr'] * x['volume'], reverse=True)
        return [pair['symbol'] for pair in pairs[:config['max_pairs']]]
    except Exception as e:
        logging.error(f"Error selecting trading pairs: {e}")
        return []

def detect_anomalies(df):
    """Обнаружение аномалий."""
    price_change = df['close'].pct_change()
    if price_change.iloc[-1] > 0.1:  # Например, изменение цены > 10%
        return True
    return False

def backtest_strategy(df):
    """Бэктестинг стратегии."""
    try:
        # Пример простого бэктеста
        initial_balance = 1000  # Начальный баланс
        balance = initial_balance
        for i in range(len(df) - 1):
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            if next_price > current_price:
                balance += balance * 0.01  # Прибыль 1%
            else:
                balance -= balance * 0.01  # Убыток 1%
        return balance
    except Exception as e:
        logging.error(f"Error backtesting strategy: {e}")
        return None

def trade_bot():
    """Основная логика бота."""
    max_daily_loss = config['max_daily_loss']
    daily_loss = 0
    while True:
        try:
            # Получение баланса
            balance = get_balance()
            if balance is None:
                time.sleep(config['check_interval'])
                continue

            # Проверка дневного убытка
            if daily_loss >= max_daily_loss:
                logging.warning("Daily loss limit reached. Pausing trading.")
                send_telegram_message("Daily loss limit reached. Pausing trading.")
                time.sleep(3600)  # Пауза на 1 час
                continue

            # Выбор торговых пар
            trading_pairs = select_trading_pairs()
            logging.info(f"Selected trading pairs: {trading_pairs}")
            send_telegram_message(f"Selected trading pairs: {trading_pairs}")

            for symbol in trading_pairs:
                # Получение исторических данных
                df = get_historical_data(symbol)
                if df is None:
                    continue

                # Проверка аномалий
                if detect_anomalies(df):
                    logging.warning(f"Anomaly detected for {symbol}. Skipping.")
                    continue

                # Автоматическая настройка параметров
                params = auto_adjust_parameters(df)
                logging.info(f"Auto-adjusted parameters for {symbol}: {params}")
                send_telegram_message(f"Auto-adjusted parameters for {symbol}: {params}")

                # Оптимизация гиперпараметров
                best_params = optimize_hyperparameters(df)
                logging.info(f"Optimized hyperparameters for {symbol}: {best_params}")
                send_telegram_message(f"Optimized hyperparameters for {symbol}: {best_params}")

                # Расчет размера позиции
                position_size = calculate_position_size(balance, config['risk_per_trade'], params['stop_loss'])
                logging.info(f"Position size for {symbol}: {position_size}")

                # Рассчет индикаторов
                rsi = calculate_rsi(df['close'], config['rsi_period'])
                macd, signal = calculate_macd(df)
                upper_band, lower_band = calculate_bollinger_bands(df)

                # Логика торговли
                if rsi > params['rsi_overbought'] and macd > signal and df['close'].iloc[-1] > upper_band:
                    logging.info(f"RSI indicates overbought for {symbol}, selling!")
                    order = place_order_with_stop_loss(symbol, 'sell', position_size, params['stop_loss'])
                    if order:
                        daily_loss += calculate_loss(order)  # Учет убытка
                elif rsi < params['rsi_oversold'] and macd < signal and df['close'].iloc[-1] < lower_band:
                    logging.info(f"RSI indicates oversold for {symbol}, buying!")
                    order = place_order_with_stop_loss(symbol, 'buy', position_size, params['stop_loss'])
                    if order:
                        daily_loss += calculate_loss(order)  # Учет убытка

                time.sleep(10)  # Задержка между символами
            time.sleep(config['check_interval'])
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            send_telegram_message(f"Unexpected error: {e}")
            time.sleep(config['check_interval'])

def run_flask():
    """Запуск веб-интерфейса."""
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Запуск бота и веб-интерфейса в отдельных потоках
    threading.Thread(target=trade_bot).start()
    threading.Thread(target=run_flask).start()
    threading.Thread(target=send_profit_message).start()  # Добавляем эту строку