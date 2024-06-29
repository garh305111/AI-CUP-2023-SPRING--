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
