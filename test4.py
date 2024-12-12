import pandas as pd
import numpy as np

# サンプルデータを作成
data = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06'],
    'price': [100, np.nan, np.nan, 130, np.nan, 150],
    'origin': ['A', np.nan, np.nan, np.nan, np.nan, 'B']
})

# 価格のデータを線形補完
data['price'] = data['price'].interpolate(method='linear')

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
data = custom_fillna(data, 'origin', 'date')

# 結果を表示
print(data)
