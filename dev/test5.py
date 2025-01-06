import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータを作成
data = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=365*5, freq='D'),
    'price': np.random.rand(365*5)
})

# 年が変わるインデックスを特定
data['year'] = data['date'].dt.year
year_change_indices = data[data['year'].diff() != 0].index

# 予測値を可視化
plt.figure(figsize=(15, 5))

# トレーニングデータと検証データの予測結果をまとめて表示
plt.plot(range(len(data['price'])), data['price'], label='Actual Price')

# 年が変わるところに縦線を追加
for idx in year_change_indices:
    plt.axvline(x=idx, color='r', linestyle='--', linewidth=1)

plt.legend()
plt.show()
