import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from ta.volume import MFIIndicator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas_ta as ta

# Initialize MT5 connection
mt5.initialize()

# Define the symbol and timeframe
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M15
start = datetime(2023, 1, 1)
end = datetime(2024, 1, 1)

# Fetch historical data
rates = mt5.copy_rates_range(symbol, timeframe, start, end)
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
data.set_index('time', inplace=True)

mt5.shutdown()

# Alligator Indicator
def alligator(data, jaw_length=13, teeth_length=8, lips_length=5, jaw_offset=8, teeth_offset=5, lips_offset=3):
    data['jaw'] = data['close'].rolling(window=jaw_length).mean().shift(jaw_offset)
    data['teeth'] = data['close'].rolling(window=teeth_length).mean().shift(teeth_offset)
    data['lips'] = data['close'].rolling(window=lips_length).mean().shift(lips_offset)
    return data

# Stochastic Oscillator
def stochastic(data, k_period=14, d_period=3):
    data['low_k'] = data['low'].rolling(window=k_period).min()
    data['high_k'] = data['high'].rolling(window=k_period).max()
    data['%K'] = 100 * (data['close'] - data['low_k']) / (data['high_k'] - data['low_k'])
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    return data

# ATR Indicator
def atr(data, atr_period=14):
    data['hl'] = data['high'] - data['low']
    data['hc'] = abs(data['high'] - data['close'].shift(1))
    data['lc'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['hl', 'hc', 'lc']].max(axis=1)
    data['atr'] = data['tr'].rolling(window=atr_period).mean()
    return data

# Fractals
def fractals(data, n=2):
    data['fractal_up'] = data['high'][(data['high'] > data['high'].shift(1)) & (data['high'] > data['high'].shift(-1))]
    data['fractal_down'] = data['low'][(data['low'] < data['low'].shift(1)) & (data['low'] < data['low'].shift(-1))]
    return data

# Machine Learning Enhanced MFI
def ml_mfi(data, mfi_length=14, train_length=300, iterations=5):
    mfi = MFIIndicator(high=data['high'], low=data['low'], close=data['close'], volume=data['tick_volume'], window=mfi_length).money_flow_index()
    data['mfi'] = mfi

    def kmeans_adjust(data, iterations, train_length):
        kmeans = KMeans(n_clusters=3)
        for i in range(iterations):
            sample = data['mfi'].iloc[-train_length:]
            kmeans.fit(sample.values.reshape(-1, 1))
            centers = sorted(kmeans.cluster_centers_.flatten())
            data['overbought'], data['neutral'], data['oversold'] = centers[2], centers[1], centers[0]
            data['mfi'] = data['mfi'].apply(lambda x: (x - centers[0]) / (centers[2] - centers[0]) * 100)

        return data

    data = kmeans_adjust(data, iterations, train_length)
    return data

# Apply Indicators and Strategy Logic
def apply_strategy(data, params):
    jaw_length, teeth_length, lips_length, jaw_offset, teeth_offset, lips_offset, k_period, d_period, atr_period, atr_multiplier = params

    # Apply indicators
    data = alligator(data, jaw_length=int(jaw_length), teeth_length=int(teeth_length), lips_length=int(lips_length), jaw_offset=int(jaw_offset), teeth_offset=int(teeth_offset), lips_offset=int(lips_offset))
    data = stochastic(data, k_period=int(k_period), d_period=int(d_period))
    data = atr(data, atr_period=int(atr_period))
    data = fractals(data)
    data = ml_mfi(data)

    # Define strategy conditions
    data['buy_signal'] = (data['close'] > data['lips']) & (data['mfi'] < 20) & (data['%K'] > data['%D'])
    data['sell_signal'] = (data['close'] < data['lips']) & (data['mfi'] > 80) & (data['%K'] < data['%D'])

    # ATR trailing stop loss
    data['trail_stop'] = data['atr'] * atr_multiplier

    return data

data = apply_strategy(data, [13, 8, 5, 8, 5, 3, 14, 3, 14, 1.75])

# Backtesting with Risk Management
def backtest(data, initial_balance=10000):
    balance = initial_balance
    equity = initial_balance
    position = 0
    entry_price = 0
    risk_per_trade = 0.01  # 1% of equity
    take_profit_multiplier = 3  # 3R trading strategy

    for i in range(len(data)):
        if data['buy_signal'].iloc[i] and position == 0:
            position = 1
            entry_price = data['close'].iloc[i]
            risk_amount = equity * risk_per_trade
            stop_loss = entry_price - data['trail_stop'].iloc[i]
            take_profit = entry_price + (entry_price - stop_loss) * take_profit_multiplier

        elif data['sell_signal'].iloc[i] and position == 1:
            balance += (data['close'].iloc[i] - entry_price) * (equity / entry_price)
            position = 0

        elif position == 1:
            if data['close'].iloc[i] < stop_loss:
                balance -= risk_amount
                position = 0
            elif data['close'].iloc[i] > take_profit:
                balance += risk_amount * take_profit_multiplier
                position = 0

        equity = balance

    return balance

final_balance = backtest(data)
print(f"Final Balance: {final_balance}")

# Performance Metrics
def calculate_performance_metrics(data):
    close_prices = data['close']

    metrics = {
        'CAGR': ta.cagr(close_prices),
        'Calmar Ratio': ta.calmar_ratio(close_prices),
        'Downside Deviation': ta.downside_deviation(close_prices),
        'Log Max Drawdown': ta.log_max_drawdown(close_prices),
        'Max Drawdown': ta.max_drawdown(close_prices),
        'Pure Profit Score': ta.pure_profit_score(close_prices),
        'Sharpe Ratio': ta.sharpe_ratio(close_prices),
        'Sortino Ratio': ta.sortino_ratio(close_prices),
        'Volatility': ta.volatility(close_prices)
    }

    return metrics

metrics = calculate_performance_metrics(data)
print(f"Performance Metrics: {metrics}")

# Visualization
def plot_strategy(data, final_balance, metrics):
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(data.index, data['jaw'], label='Jaw')
    plt.plot(data.index, data['teeth'], label='Teeth')
    plt.plot(data.index, data['lips'], label='Lips')
    plt.scatter(data.index, data['fractal_up'], label='Fractal Up', marker='^', color='green')
    plt.scatter(data.index, data['fractal_down'], label='Fractal Down', marker='v', color='red')
    
    # Add final balance and performance metrics to the plot
    plt.text(data.index[-1], data['close'].max(), f'Final Balance: ${final_balance:.2f}', fontsize=12, verticalalignment='top')
    metrics_text = "\n".join([f"{key}: {value:.2f}" for key, value in metrics.items()])
    plt.text(data.index[-1], data['close'].max() - (data['close'].max() * 0.1), metrics_text, fontsize=10, verticalalignment='top')
    
    plt.title('Trading Strategy Visualization')
    plt.legend()
    plt.show()

plot_strategy(data, final_balance, metrics)
