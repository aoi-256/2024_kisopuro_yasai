import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

if __name__ == '__main__':
    np.random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # データの読み込み
    data = pd.read_csv('negi_data.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    # 特徴量とターゲットの分離
    features = data[['date', 'amount']]
    target = data['price']

    # データの標準化
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features[['amount']])
    target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))

    # シーケンスデータの作成
    seq_length = 30  # 30日間のデータを使用
    X, y = create_sequences(features_scaled, seq_length)

    # データのトレーニングセットと検証セットへの分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # モデルの構築
    model = LSTMModel(input_dim=1, hidden_dim=50, output_dim=1, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    num_epochs = 100
    batch_size = 16
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

    # トレーニングデータの予測結果
    plt.plot(range(len(target[:len(train_predictions)])), target[:len(train_predictions)], label='Actual Train Price')
    plt.plot(range(len(train_predictions)), train_predictions, label='Predicted Train Price')

    # 検証データの予測結果
    val_start_idx = len(target[:len(train_predictions)])
    plt.plot(range(val_start_idx, val_start_idx + len(val_predictions)), target[val_start_idx:val_start_idx + len(val_predictions)], label='Actual Val Price')
    plt.plot(range(val_start_idx, val_start_idx + len(val_predictions)), val_predictions, label='Predicted Val Price')

    plt.legend()
    plt.show()
