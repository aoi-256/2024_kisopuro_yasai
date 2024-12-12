import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# サンプルデータを作成
data = ['要素1_要素2', '要素1', '要素1_要素2_要素3', '要素2_要素3']
data = [elem.split('_') for elem in data]

# 各要素を列として扱うためにデータフレームに変換
df = pd.DataFrame(data)

# Nanを埋める（もし存在する場合）
df = df.fillna('')

# OneHotEncoderを使用してエンコード
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(df.apply(lambda x: ' '.join(x), axis=1).str.get_dummies(sep=' '))

# 結果の確認
print(encoded_data)
