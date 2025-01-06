'''
test7.py

気候データ込みでのデータ取得と前処理について
（スケール調整はやっていない）
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('negi_data.csv', encoding='shift_jis')
weather = pd.read_csv("negi_weather.csv")

# 日付をdatetime形式に変換
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')

''' データの補完 '''

# 20060101から全ての日付を含むデータフレームを作成
full_date_range = pd.date_range(start='2005-12-31', end=data['date'].max())
full_data = pd.DataFrame(full_date_range, columns=['date'])

# 元のデータと結合
data = pd.merge(full_data, data, on='date', how='left')

# 数値データの線形補完
data['price'] = data['price'].interpolate(method='linear')
data['amount'] = data['amount'].interpolate(method='linear')

# 産地データの補完
data['area'] = data['area'].ffill()

''' 産地データの処理 '''

# area列を処理
area = [elem.split('_') for elem in data['area']]

# 各要素を列として扱うためにデータフレームに変換
area_data = pd.DataFrame(area)

# '各地'を空文字列に置き換え、Nanを埋める
#area_data = area_data.replace('各地', '')
area_data = area_data.fillna('')

# 分割したデータのi番目の要素をdata['area_i']に格納
data['area_1'] = area_data[0]
data['area_2'] = area_data[1]
data['area_3'] = area_data[2]

encoder = OneHotEncoder(sparse_output=False)

# area_1のエンコード
encoded_area_1 = encoder.fit_transform(data[['area_1']]).astype(int)
data['area_1'] = [''.join(map(str, row)) for row in encoded_area_1]

# area_2のエンコード
encoded_area_2 = encoder.fit_transform(data[['area_2']]).astype(int)
data['area_2'] = [''.join(map(str, row)) for row in encoded_area_2]

# area_3のエンコード
encoded_area_3 = encoder.fit_transform(data[['area_3']]).astype(int)
data['area_3'] = [''.join(map(str, row)) for row in encoded_area_3]

''' 余分なデータの削除 '''

# 20051231のデータを削除
data = data[data['date'] >= '2006-01-01']
wether = weather[weather['date'] >= '2006-01-01']

# 20230101のデータを削除
data = data[data['date'] < '2023-01-01']
wether = wether[wether['date'] < '2023-01-01']

# areaのデータを削除（使用しないため）

''' 処理したデータの合成 '''

data = data.drop(columns=['area'])

data['date'] = pd.to_datetime(data['date']) 
weather['date'] = pd.to_datetime(weather['date']) 
train_data = pd.merge(data, weather, on='date')

train_data.to_csv('new_negi_data.csv', index=False)

