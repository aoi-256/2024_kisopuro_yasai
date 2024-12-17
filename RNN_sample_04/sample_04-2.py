import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. データの生成
def generate_sine_wave(seq_length, num_samples, num_features=1):
    """
    正弦波データを生成します。
    
    Args:
        seq_length (int): 一つのシーケンスの長さ（タイムステップ数）。
        num_samples (int): 全体のサンプル数。
        num_features (int): 特徴量の数（デフォルトは1）。
        
    Returns:
        X: 特徴量データ (num_samples, seq_length, num_features)
        y: ターゲットデータ (num_samples, 1)
    """
    X = []
    y = []
    # 時間軸を生成
    for i in range(len(num_samples)):
        # 複数の正弦波や他の特徴量を追加することも可能
        t = np.linspace(0, 2 * np.pi, seq_length + 1)
        sine_wave = np.sin(t + i * 0.1)  # 時系列に変化を加える
        # 特徴量を増やす場合はここで追加
        # 例: 他の正弦波や余弦波、ランダムノイズなど
        features = [sine_wave[:-1]]  # 現在のタイムステップ
        if num_features > 1:
            # 追加の特徴量を追加
            for n in range(1, num_features):
                features.append(np.sin(t[:-1] + i * 0.1 + n))
        features = np.array(features).T  # (seq_length, num_features)
        X.append(features)
        y.append(sine_wave[-1])  # 次の値を予測
    return np.array(X), np.array(y)

# パラメータ設定
SEQ_LENGTH = 50  # タイムステップ数
NUM_SAMPLES = 1000  # サンプル数
NUM_FEATURES = 1  # 特徴量の数（簡単に拡張可能）

# データ生成
X, y = generate_sine_wave(SEQ_LENGTH, NUM_SAMPLES, NUM_FEATURES)

# データのスケーリング
scaler_X = MinMaxScaler()
# Reshape X to (samples * timesteps, features) for scaling
X_reshaped = X.reshape(-1, NUM_FEATURES)
X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# トレーニングデータとテストデータに分割
train_size = int(0.8 * NUM_SAMPLES)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# モデルの構築
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, NUM_FEATURES)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# モデルの訓練
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 予測
y_pred = model.predict(X_test)

# スケールを元に戻す
y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# 結果のプロット
plt.figure(figsize=(12,6))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title('Sine Wave Prediction using LSTM')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.show()