import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('negi_data.csv', encoding='shift_jis')

# 許容できる最小値と最大値を 定義
lower_bound = 250
upper_bound = 3000

# 外れ値を処理する関数を定義
def handle_outliers(series, lower_bound, upper_bound):
    series = series.apply(lambda x: lower_bound if x < lower_bound else x)
    series = series.apply(lambda x: upper_bound if x > upper_bound else x)
    return series

# price列の外れ値を処理
data['price'] = handle_outliers(data['price'], lower_bound, upper_bound)

# 結果を確認
print(data)
