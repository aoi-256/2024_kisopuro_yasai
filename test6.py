import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

if __name__ == '__main__':
    # データの読み込み
    data = pd.read_csv('raw_negi_data.csv', encoding='shift_jis')

    # 価格のデータを線形補完
    data['price'] = data['price'].interpolate(method='linear')

    print(data)

    # 産地のデータを特定のルールに基づいて補完する関数
    def custom_fillna(df, value_col, date_col):
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

    # 産地のデータを補完
    data = custom_fillna(data, 'area', 'date')

    """ 価格データの処理 """

    # 許容できる最小値と最大値を定義
    lower_bound = 500
    upper_bound = 5000

    # 外れ値を処理する関数を定義
    def handle_outliers(series, lower_bound, upper_bound):
        series = series.apply(lambda x: lower_bound if x < lower_bound else x)
        series = series.apply(lambda x: upper_bound if x > upper_bound else x)
        return series

    # price列の外れ値を処理
    data['price'] = handle_outliers(data['price'], lower_bound, upper_bound)

    """ 日付データの処理 """

    data['date'] = pd.to_datetime(data['date'])

    """ 産地データの処理 """

    area = [elem.split('_') for elem in data['area']]

    # 各要素を列として扱うためにデータフレームに変換
    area_data = pd.DataFrame(area)

    # データがないところと各地の項を空にする
    #area_data = area_data.replace('各地', '')
    area_data = area_data.fillna('')

    # 分割したデータの1番目の要素をdata['area_1']に格納
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



    print(data.head())