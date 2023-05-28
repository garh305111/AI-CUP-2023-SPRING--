import pandas as pd
import numpy as np
import librosa
import os
import sys
import re
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


class SoftVoter(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers):
        self.classifiers = classifiers
        
    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)
    
    def predict(self, X):
        predictions = []
        for classifier in self.classifiers:
            predictions.append(classifier.predict(X).reshape(-1, 1))
        
        final_predictions = []
        for samples in np.concatenate(predictions, axis=1):
            vote_counts = Counter(samples)
            majority_vote = vote_counts.most_common(1)[0][0]
            final_predictions.append(majority_vote)
        
        return np.array(final_predictions)

def weight_calc(df):
    return df["Disease category"].value_counts().map(lambda x: 1/(x/len(df["Disease category"]))).to_dict()

def custom_cross_val_score(model, X, y, cv=5):
    scores = []
    n_samples = len(X)
    fold_size = n_samples // cv

    for i in range(cv):
        start = i * fold_size
        end = (i+1) * fold_size
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)
        X_val = X[start:end]
        y_val = y[start:end]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        score = recall_score(y_val, y_pred, average='macro')
        scores.append(score)

    return scores

def stacking_ensemble_fit(X_train, y_train, X_test, num_layers, weights, model_save_path):

    models = [
        RandomForestClassifier(criterion="gini", class_weight=weights),
        ExtraTreesClassifier(criterion="gini", class_weight=weights),
        XGBClassifier(use_label_encoder=True),
        CatBoostClassifier(class_weights=weights),
        LGBMClassifier(class_weight=weights),
        KNeighborsClassifier(weights="uniform")
    ]

    base_model_predictions_train = []
    base_model_predictions_test = []
    X_train_stack = X_train
    X_test_stack = X_test

    for _ in range(num_layers):
        layer_predictions_train = []
        layer_predictions_test = []

        for model in models:
            model.fit(X_train_stack, y_train)
            train_predictions = model.predict(X_train_stack)
            test_predictions = model.predict(X_test_stack)
            layer_predictions_train.append(train_predictions)
            layer_predictions_test.append(test_predictions)

        base_model_predictions_train.extend(layer_predictions_train)
        base_model_predictions_test.extend(layer_predictions_test)
        X_train_stack = np.column_stack((X_train_stack, *layer_predictions_train))
        X_test_stack = np.column_stack((X_test_stack, *layer_predictions_test))

    # 建立 SoftVoter 分類器
    soft_voter = SoftVoter(models)
    cv_scores = custom_cross_val_score(soft_voter, X_train_stack, y_train, cv=10)
    print("validation results:", np.mean(cv_scores))

    # 在整個訓練集上擬合最終模型
    soft_voter.fit(X_train_stack, y_train)

    # 將模型保存到指定位置
    with open(model_save_path, 'wb') as file:
        pickle.dump(soft_voter, file)

    # 在測試集上進行預測
    test_predictions = soft_voter.predict(X_test_stack)

    return test_predictions


isnan_columns = lambda df, column: True if df[column].map(lambda x: np.isnan(x)).value_counts()[False] != len(df[column]) else False

if "__name__" == "main":
    if len(sys.argv) != 7:
        print("錯誤的命令行參數！")
        print("使用方法: python3 script.py <train_audio_file_path> <train_table_path> <model_storage_path> "
            "<test_audio_file_path> <test_table_path> <prediction_storage_path>")
        sys.exit(1)

    train_audio_file_path = sys.argv[1]
    train_table_path = sys.argv[2]
    model_storage_path = sys.argv[3]
    test_audio_file_path = sys.argv[4]
    test_table_path = sys.argv[5]
    prediction_storage_path = sys.argv[6]


    df = pd.read_csv(train_table_path)
    df = df.set_index("ID")
    weights = weight_calc(df)
    df["Voice handicap index - 10"].hist() #VHI-10嗓音障礙指標 0-40
    print("nan numbers:", df["Voice handicap index - 10"].isna().sum())
    print("median:", df["Voice handicap index - 10"].median())

    idx = df[df["Voice handicap index - 10"].map(lambda x: np.isnan(x)) == True].index
    df.loc[idx, "Voice handicap index - 10"] = df["Voice handicap index - 10"].median()

    categorical_columns = ['Sex', 'Smoking', 'Diurnal pattern', 'Onset of dysphonia ', 'Occupational vocal demand']
    df_trans = pd.get_dummies(df, columns=categorical_columns)
    df_trans['Age'] = df['Age'] / 100
    df_trans['Drinking'] = df['Drinking'] / 2
    df_trans['frequency'] = df['frequency'] / 3
    df_trans['Noise at work'] = (df['Noise at work'] - 1) / 2
    df_trans['Occupational vocal demand'] = (4 - df['Occupational vocal demand']) / 3
    df_trans['Voice handicap index - 10'] = df['Voice handicap index - 10'] / 40
    df_trans.loc[df_trans[df_trans["PPD"].isna() == True].index.tolist(), ["PPD"]] = df_trans[df_trans["PPD"].isna() == True]["PPD"].map(lambda x: 1)

    audio_df = pd.DataFrame()
    for file in os.listdir(train_audio_file_path):
        print(file)
        y, sr = librosa.load(f'{train_audio_file_path}/{file}')
        mfccs = np.median(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200).T, axis = 0)
        mfccs = np.concatenate((mfccs, np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200).T, axis=0)))
        print(mfccs.shape)
        audio_df.loc[re.sub('.wav', '', file), [i for i in range(len(mfccs))]] = mfccs
    df_trans_with_audio = pd.concat([df_trans, audio_df.loc[df_trans.index, :]], axis=1)
    X_train = df_trans_with_audio.drop(columns=["Disease category"]).values
    y_train = df_trans_with_audio["Disease category"].values
    #----------------------------------------------------------------------------------------

    df_test = pd.read_csv(test_table_path)
    df_test = df_test.set_index("ID")
    idx = df_test[df_test["Voice handicap index - 10"].map(lambda x: np.isnan(x)) == True].index
    df_test.loc[idx, "Voice handicap index - 10"] = df["Voice handicap index - 10"].median()
    df_trans_test = pd.get_dummies(df_test, columns=categorical_columns)

    df_trans_test['Age'] = df_test['Age'] / 100
    df_trans_test['Drinking'] = df_test['Drinking'] / 2
    df_trans_test['frequency'] = df_test['frequency'] / 3
    df_trans_test['Noise at work'] = (df_test['Noise at work'] - 1) / 2
    df_trans_test['Occupational vocal demand'] = (4 - df_test['Occupational vocal demand']) / 3
    df_trans_test['Voice handicap index - 10'] = df_test['Voice handicap index - 10'] / 40
    df_trans_test.loc[df_trans_test[df_trans_test["PPD"].isna() == True].index.tolist(), ["PPD"]] = df_trans_test[df_trans_test["PPD"].isna() == True]["PPD"].map(lambda x: 1)
    audio_df = pd.DataFrame()
    for file in os.listdir(test_audio_file_path):
        print(file)
        y, sr = librosa.load(f'{test_audio_file_path}/{file}')
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200).T, axis = 0)

        audio_df.loc[re.sub('.wav', '', file), [i for i in range(len(mfccs))]] = mfccs
    df_trans_test_with_audio = pd.concat([df_trans_test, audio_df.loc[df_trans_test.index, :]], axis=1)
    X_test = df_trans_test_with_audio.values

    print("Validation Macro Recall Scores:")
    test_pred = stacking_ensemble_fit(X_train, y_train, X_test, 3, weights, model_storage_path)
    pd.DataFrame(test_pred).to_csv(prediction_storage_path, header=False, columns=False)
