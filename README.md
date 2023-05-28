# AI-CUP-2023-SPRING

## Project Title

AI CUP 2023 春季賽多模態病理嗓音分類競賽

### Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

### Installation

安裝所需的套件

```bash
pip install pandas numpy librosa matplotlib scikit-learn xgboost catboost lightgbm
```

## Usage

This project provides functionality for audio file analysis and prediction using machine learning models. Below are the instructions on how to use the project, including the available scripts and command-line options.

### Script: `model_building.py`

This script is used for building and training machine learning models based on audio files and a table of features.

Command-line usage:
```
python3 model_building.py <audio_file_path> <table_path> <model_storage_path>
```

- `<audio_file_path>`: The file path to the input audio file.
- `<table_path>`: The file path to the table containing the audio features.
- `<model_storage_path>`: The directory path where the trained model will be stored.

### Script: `predict.py`

This script is used for making predictions on new audio files using the trained models.

Command-line usage:
```
python3 predict.py <audio_file_path> <table_path> <prediction_storage_path>
```

- `<audio_file_path>`: The file path to the input audio file for prediction.
- `<table_path>`: The file path to the table containing the audio features.
- `<prediction_storage_path>`: The directory path where the prediction output file will be stored.

Please note that you need to have the trained models available in the specified `<model_storage_path>` for the prediction script to work correctly.

## Output

The output of the prediction script (`predict.py`) will be a prediction file stored in the specified `<prediction_storage_path>`. This file will contain the predicted results for the input audio file.

Make sure to provide the correct input file paths and storage paths when using the scripts to ensure the desired functionality of the project.


### License

This project is licensed under the MIT License. See the LICENSE file for more information.
