# galaxy_cnn

## 環境設定
### 環境
- Python3

### 設定
- pip install -r requirements.txt


## 使い方
- python galaxy_learning_multi_channels_files_input.py [入力ファイルパス]

### 入力ファイルの形式
- CSVで構造は以下の通り（３バンドの画像を使用）
- 天体のカタログID, 等級（今は使っていない）, 1つめのバンドの画像パス, 2つめのバンドの画像パス, 3つめのバンドの画像パス, 正解ラベル(0 or 1)
- 入力ファイルの画像のパスを自分の使いたいデータのパスにすることで，任意の画像で学習ができる
