# iPhoneヘルスケアアプリ歩数読取ツール

### Introduction

iPhoneヘルスケアアプリ歩数読取ツール（以下、本ツール）は、iPhoneのヘルスケアアプリに表示される歩数の棒グラフを画像解析し、日々の歩数をExcelファイルとして書き出すツールです。

<div align='center'>
<img src="./data_sample/iPhone11Pro.png" title='歩数の棒グラフのイメージ' height="450" />
</div>

### Requirement
* Windows OS (64 bit) (Windows 10で動作確認済)
* Python 3.6 (3.6.13で動作確認済)
* Pythonパッケージ
  ```shell
  pip install -r requirements.txt
  ```
* Tesseract (64 bit) (v5.0.0-alpha.20200328で動作確認済)
  - OCRソフト
  - インストーラは https://github.com/UB-Mannheim/tesseract/wiki から入手可能

### Usage

1. Git Clone

   ```shell
   git https://github.com/HMAdachi-THK/healthcareapp-stepsreader.git
   ```

2. データ準備

  * 'data'フォルダに解析するヘルスケアアプリ画像を格納する
  
3. 実行

  * run.cmdをダブルクリックする
  * 実行結果は'output'フォルダに出力される
  
