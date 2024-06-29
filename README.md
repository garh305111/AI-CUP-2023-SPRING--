# AI-CUP-2023-SPRING

## Project Title

AI CUP 2023 春季賽多模態病理嗓音分類競賽

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

安裝所需的套件

```bash
pip install -r requirements.txt
```

## Usage

本項目提供音檔分析和預測的功能，使用機器學習模型進行操作。
以下是使用該項目的說明，包括可用的腳本和命令行選項。

Script：script.py
此腳本用於基於音檔和特徵表構建和訓練機器學習模型，並進行預測。

命令行使用方法：

```python3
python3 script.py \
  <train_audio_file_path> \
  <train_table_path> \
  <model_storage_path> \
  <test_audio_file_path> \
  <test_table_path> \
  <prediction_storage_path>
```
<audio_file_path>：輸入音檔的文件路徑。 \
<table_path>：特徵表的文件路徑。 \
<model_storage_path>：訓練好的模型將存儲的目錄路徑。 \
<audio_file_path>：進行預測的輸入音檔的文件路徑。 \
<table_path>：特徵表的文件路徑。 \
<prediction_storage_path>：預測結果將存儲的目錄路徑。 \
請注意，在使用預測腳本之前，需要確保訓練好的模型位於指定的 <prediction_storage_path> 中。

輸出
script.py的輸出將是存儲在指定的 <prediction_storage_path> 中的預測檔案。\
該文件將包含對輸入音檔的預測結果。


## License

This project is licensed under the MIT License. See the LICENSE file for more information.
