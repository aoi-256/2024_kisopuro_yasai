# 2024_kisopuro_yasai

野菜の価格をRNNを用いて予測するコードです

データはコンテストのものを使用しています

[コンテストURL](https://competition.nishika.com/competitions/yasai_2024winter/summary)

## 各ファイルについて

### negi_data.csv

・コンテストのデータからねぎに関する項目を抽出し、並び替えたもの

#### データの補完について

・市場データがない日については線形補完している

・産地データは、欠損区間の前方と後方の区間を半分ずつコピーしている

### RNN_sample_01 

・三角関数関数の予測

・精度良好につき開発終了

### RNN_sample_02 

・価格予測への応用を試みた

・説明変数が１つのみであり、精度が出なく開発中止(未完成)

### RNN_sample_03 

・多数の説明変数にも対応した

・現在開発中

![negi_data-amount_01](https://github.com/user-attachments/assets/58c31738-0c87-44db-9f92-ad0a5fb6434a)

### RNN_sample_04 

・気候データの導入を行う

#### 扱う気候データについて

| データ名   | 内容          |
| ----      | ----          |
| mean_temp | １日の平均気温 |
| sum_rain  | １日の降水量   |
| sum_time  | １日の日照時間 |
