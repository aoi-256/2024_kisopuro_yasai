import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# サンプルデータを作成
data = pd.read_csv('negi_data.csv', encoding='shift_jis')

# area列を処理
area = [elem.split('_') for elem in data['area']]

# 各要素を列として扱うためにデータフレームに変換
area_data = pd.DataFrame(area)

# Nanを埋める（もし存在する場合）
area_data = area_data.fillna('')

# 各要素の配列を出力
elements = area_data.values.T.tolist()

# 分割したデータの1番目の要素をdata['area_1']に格納
data['area_1'] = area_data[0]
data['area_2'] = area_data[1]
data['area_3'] = area_data[2]

encoder = OneHotEncoder(sparse_output=False)
for col in ['area_1', 'area_2', 'area_3']: 
    encoded = encoder.fit_transform(data[[col]]) 
    encoded_df = pd.DataFrame(encoded, columns=[f'{col}_{i}' for i in range(encoded.shape[1])]) 
    data = pd.concat([data, encoded_df], axis=1)

# 結果を表示
print(data.head())


