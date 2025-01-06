import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# データの読み込み
data = pd.read_csv('negi_data.csv', encoding='shift_jis')

# area列を処理
area = [elem.split('_') for elem in data['area']]

# 各要素を列として扱うためにデータフレームに変換
area_data = pd.DataFrame(area)

# '各地'を空文字列に置き換え、Nanを埋める
area_data = area_data.replace('各地', '')
area_data = area_data.fillna('')

# 分割したデータの1番目の要素をdata['area_1']に格納
data['area_1'] = area_data[0]
data['area_2'] = area_data[1]
data['area_3'] = area_data[2]

encoder = OneHotEncoder(sparse_output=False)

# area_1のエンコード
encoded_area_1 = encoder.fit_transform(data[['area_1']]).astype(int)
data['encoded_area_1'] = [''.join(map(str, row)) for row in encoded_area_1]

# area_2のエンコード
encoded_area_2 = encoder.fit_transform(data[['area_2']]).astype(int)
data['encoded_area_2'] = [''.join(map(str, row)) for row in encoded_area_2]

# area_3のエンコード
encoded_area_3 = encoder.fit_transform(data[['area_3']]).astype(int)
data['encoded_area_3'] = [''.join(map(str, row)) for row in encoded_area_3]

# 結果を表示
print(data[['encoded_area_1', 'encoded_area_2', 'encoded_area_3']].head())
