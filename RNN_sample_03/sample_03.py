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

class EarlyStopping:
    """earlystoppingクラス"""

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        """引数：最小値の非更新数カウンタ、表示設定、モデル格納path"""
        self.patience = patience    #設定ストップカウンタ
        self.verbose = verbose      #表示の有無
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.inf  #前回のベストスコア記憶用
        self.path = path            #ベストモデル格納path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            if self.verbose:  #表示を有効にした場合は経過を表示
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True

                return True #self.early_stop = Trueがうまく動作しないので応急処置
            
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            self.checkpoint(val_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, model):
        '''ベストスコア更新時に実行されるチェックポイント関数'''
        if self.verbose:  #表示を有効にした場合は、前回のベストスコアからどれだけ更新したか？を表示
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  #ベストモデルを指定したpathに保存
        self.val_loss_min = val_loss  #その時のlossを記録する

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]

        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y[0]  # 価格データだけを選択
        
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == '__main__':

    #params

    seq_length = 30
    num_epochs = 1000
    batch_size = 16
    test_data_size = 0.2 #学習データと評価用のデータの比率
    lr = 0.0001 #学習率
    lower_bound = 500 #許容できる最小値と最大値を定義
    upper_bound = 5000


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # データの読み込み
    data = pd.read_csv('negi_data.csv', encoding='shift_jis')

    """ 価格データの処理 """

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
    
    """ 学習データの設定 """

    # 特徴量とターゲットの分離
    features = data[['date', 'amount', 'area_1', 'area_2', 'area_3']]
    target = data['price']

    # データの標準化
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features[['amount', 'area_1', 'area_2', 'area_3']])
    target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

    # シーケンスデータの作成
    # seq_length = 30(paramに移動)
    X, y = create_sequences(features_scaled, seq_length)

    # データのトレーニングセットと検証セットへの分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_data_size, shuffle=False)

    # モデルの構築
    model = LSTMModel(input_dim=4, hidden_dim=100, output_dim=1, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    es = EarlyStopping(patience=10, verbose=True)

    def train_model(model, X_train, y_train, X_val, y_val, num_epochs, batch_size):
        train_loss_history = []
        val_loss_history = []
        
        for epoch in range(num_epochs):

            model.train()
            epoch_loss = 0
            x_, t_ = shuffle(X_train, y_train)
            for i in range(0, len(X_train), batch_size):
                X_batch = torch.tensor(x_[i:i+batch_size], dtype=torch.float32).to(device)
                y_batch = torch.tensor(t_[i:i+batch_size], dtype=torch.float32).to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(X_train)
            train_loss_history.append(train_loss)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    X_val_batch = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
                    y_val_batch = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32).to(device)
                    outputs = model(X_val_batch)
                    val_loss += criterion(outputs, y_val_batch).item()
                    
            val_loss /= len(X_val)
            val_loss_history.append(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

            if es(val_loss, model):
                break

            

        return train_loss_history, val_loss_history
    
    # モデルのトレーニング
    #num_epochs = 1000 (paramに移動)
    #batch_size = 8(paramに移動)
    train_loss_history, val_loss_history = train_model(model, X_train, y_train, X_val, y_val, num_epochs, batch_size)


    # モデルの評価と予測
    model.eval()
    train_predictions = []
    val_predictions = []

    # トレーニングデータの予測
    for i in range(0, len(X_train), batch_size):
        X_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            batch_preds = model(X_batch)
            train_predictions.extend(batch_preds.cpu().numpy())

    # 検証データの予測
    for i in range(0, len(X_val), batch_size):
        X_val_batch = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            batch_preds = model(X_val_batch)
            val_predictions.extend(batch_preds.cpu().numpy())

    # 予測値を元のスケールに戻す
    train_predictions = scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1))
    val_predictions = scaler.inverse_transform(np.array(val_predictions).reshape(-1, 1))

    # 予測値を可視化
    plt.figure(figsize=(15, 5))

    # トレーニングデータと検証データの予測結果をまとめて表示
    plt.plot(range(len(target)), target, label='Actual Price')
    plt.plot(range(len(train_predictions)), train_predictions, label='Predicted Price (Train)')
    val_start_idx = len(train_predictions)
    plt.plot(range(val_start_idx, val_start_idx + len(val_predictions)), val_predictions, label='Predicted Price (Val)')

    plt.legend()
    plt.show()