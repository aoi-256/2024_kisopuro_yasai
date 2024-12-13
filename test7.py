'''
test7.py

気候データ込みでのデータ取得と前処理について
（スケール調整はやっていない）
'''

import pandas as pd
import numpy as np

def area_data_fill(df, value_col, date_col):

    is_nan = df[value_col].isna()
    nan_indices = np.where(is_nan)[0]
    
    if len(nan_indices) == 0:
        return df

    start_idx = nan_indices[0]
    end_idx = nan_indices[-1] + 1

    # 連続する欠損値の長さを数える
    nan_length = end_idx - start_idx

    # 前半部分を前方補完、後半部分を後方補完
    half_length = nan_length // 2
    df[value_col].iloc[start_idx:start_idx + half_length] = df[value_col].iloc[start_idx - 1]
    df[value_col].iloc[start_idx + half_length:end_idx] = df[value_col].iloc[end_idx]

    return df

# サンプルデータを作成
data = pd.read_csv('raw_negi_data.csv', encoding='shift_jis')

# 日付をdatetime形式に変換
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')

# 20060101から全ての日付を含むデータフレームを作成
full_date_range = pd.date_range(start='2005-12-31', end=data['date'].max())
full_data = pd.DataFrame(full_date_range, columns=['date'])

# 元のデータと結合
data = pd.merge(full_data, data, on='date', how='left')

# 日付以外の値を線形補完
data['price'] = data['price'].interpolate(method='linear')
data['amount'] = data['amount'].interpolate(method='linear')

# area列の欠損値を補完する
data = area_data_fill(data, 'area', 'date')

# 20051231のデータを削除
data = data[data['date'] >= '2006-01-01']

# 結果を表示
print(data)
