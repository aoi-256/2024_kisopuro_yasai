import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 日付を2006年1月1日から何日経過したか（常に366日）に変換する関数
def days_since_2006(date_str):

    date_str = str(date_str) 
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    
    date = pd.Timestamp(year, month, day)
    start_date = pd.Timestamp(2006, 1, 1)
    days_since = (date - start_date).days + 1
    
    return days_since

# CSVファイルからデータを読み込み
data = pd.read_csv('negi_data.csv')

# 日付のデータの処理
data['date'] = data['date'].apply(days_since_2006)

# 'date' をインデックスに設定
data.set_index('date', inplace=True)

# 欠損値（飛んでいる値）を補完するために全ての日付を生成
full_index = pd.RangeIndex(start=data.index.min(), stop=data.index.max() + 1, step=1)

# 再インデックス化して欠損値を含むデータフレームを作成
data = data.reindex(full_index)

# 線形補間を使用して欠損値を補完
data['price']  = data['price'].interpolate(method='linear')
data['amount'] = data['amount'].interpolate(method='linear')

# 必要なデータの選択
newdata = data[['price', 'amount']]

newdata.to_csv('RNN_sample_data4-5.csv', index=False)

# グラフの描画
plt.figure(figsize=(10, 6))
plt.plot(newdata.index, newdata['price_delta'], marker='o', label='Price Delta')
plt.xlabel('Date Count')
plt.ylabel('Price Delta')
plt.title('Price Delta with Linear Interpolation')
plt.legend()
plt.grid(True)

# 365日ごとに縦線を追加
for i in range(0, len(newdata), 365):
    plt.axvline(x=i, color='r', linestyle='--', linewidth=0.8)

for i in range(183, len(newdata), 365):
    plt.axvline(x=i, color='b', linestyle='--', linewidth=0.8)

plt.show()
