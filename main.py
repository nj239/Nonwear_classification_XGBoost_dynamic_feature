

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             balanced_accuracy_score, roc_curve, f1_score)
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_csv(file_path, label_column):
    """
    Load and preprocess data from a single CSV file.

    Args:
        file_path (str): Path to the CSV file.
        label_column (str): Name of the column to use as the binary label.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    df = pd.read_csv(file_path)

    # Ensure the required columns exist
    if not {'Date', 'Time', label_column}.issubset(df.columns):
        raise ValueError(f"The input file must contain 'Date', 'Time', and '{label_column}' columns.")

    # Combine date and time columns into a single timestamp column
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # Filter binary label
    
    df = df[df[label_column].isin([0, 1])].dropna()

    # Extract additional features
    # convert the timestamp column to datetime
   
    # extract the minute, hour, and quarter
    df['minute'] = df['timestamp'].dt.minute
    df['hour'] = df['timestamp'].dt.hour
    df['quarter'] = df['timestamp'].dt.quarter

    # define a function to determine morning/night
    def get_time_of_day(hour):
        if hour >= 5 and hour < 12:
            return 'morning'
        elif hour >= 12 and hour < 18:
            return 'afternoon'
        else:
            return 'night'

    # extract the day of the week
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # apply the function to the hour column
    df['time_of_day'] = df['hour'].apply(get_time_of_day)

    
    # extract the day of the week
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # define a function to determine the time of day as a sine and cosine wave
    def convert_time_of_day_to_sine_cosine(time_of_day):
        if time_of_day == 'morning':
            hour_in_radians = 6 * np.pi / 12.0
        elif time_of_day == 'afternoon':
            hour_in_radians = 15 * np.pi / 12.0
        else:
            hour_in_radians = 21 * np.pi / 12.0
        return np.sin(hour_in_radians), np.cos(hour_in_radians)

    # apply the function to the time_of_day column
    df['time_of_day_sin'], df['time_of_day_cos'] = zip(*df['hour'].apply(get_time_of_day).apply(convert_time_of_day_to_sine_cosine))

    # define a function to convert day of the week to a sine and cosine wave
    def convert_day_of_week_to_sine_cosine(day_of_week):
        day_in_radians = day_of_week * np.pi / 3.5
        return np.sin(day_in_radians), np.cos(day_in_radians)

    # apply the function to the day_of_week column
    df['day_of_week_sin'], df['day_of_week_cos'] = zip(*df['day_of_week'].apply(convert_day_of_week_to_sine_cosine))


    # define a function to convert a value to a sine and cosine wave
    def convert_to_sine_cosine(value, max_value):
        value_in_radians = value * 2.0 * np.pi / max_value
        return np.sin(value_in_radians), np.cos(value_in_radians)

    # apply the function to the hour, minute, and quarter columns
    df['hour_sin'], df['hour_cos'] = zip(*df['hour'].apply(convert_to_sine_cosine, max_value=24))
    df['minute_sin'], df['minute_cos'] = zip(*df['minute'].apply(convert_to_sine_cosine, max_value=60))
    df['quarter_sin'], df['quarter_cos'] = zip(*df['quarter'].apply(convert_to_sine_cosine, max_value=4))
    df = df.dropna()
    print(df.columns)
    return df

def split_data(df, label_column, train_ratio=0.9):
    """
    Split data into training and testing sets.

    Args:
        df (pd.DataFrame): Input dataframe.
        label_column (str): Name of the column to use as the binary label.
        train_ratio (float): Ratio of data to use for training.

    Returns:
        tuple: Training and testing datasets as (X_train, y_train, X_test, y_test).
    """
    df = df.reset_index(drop=True)
    train_size = int(len(df) * train_ratio)

    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]

    features = [col for col in df.columns if col not in ['timestamp', label_column,'day_of_week', 'time_of_day', 'Date', 'Time']]#remove unnecessary timestamp related variables
 

    X_train = train_set[features].values
    y_train = train_set[label_column].values

    X_test = test_set[features].values
    y_test = test_set[label_column].values

    return X_train, y_train, X_test, y_test

def binary_array(array, threshold):
    return np.array([1 if value >= threshold else 0 for value in array])

#the temporal smoothing function
def smooth_window(arr, window):
    for i in range(int(window / 2), len(arr) - int(window / 2)):
        if sum(arr[i - int(window / 2):i + int(window / 2) + 1]) - arr[i] > int(window / 2):
            arr[i] = 1
        elif sum(arr[i - int(window / 2):i + int(window / 2) + 1]) - arr[i] < int(window / 2):
            arr[i] = 0
    return arr

def measures(y_true, y_pred):
    """
    Compute evaluation metrics.

    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.

    Returns:
        list: Evaluation metrics [sensitivity, specificity, balanced_accuracy, f1_score, ppv, npv].
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2
    f1 = f1_score(y_true, y_pred)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return [sensitivity, specificity, balanced_accuracy, f1, ppv, npv]

def evaluate_predictions(model, X_train, y_train, X_test, y_test):
    """
    Evaluate predictions with metrics and smoothing.

    Args:
        model: Trained model.
        X_train, y_train: Training data.
        X_test, y_test: Testing data.

    Returns:
        pd.DataFrame: Metrics for training and testing sets.
    """
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    thresholdOpt = 0.55 #the threshold found best for our study
    y_pred_train = binary_array(y_prob_train, thresholdOpt)
    y_pred_test = binary_array(y_prob_test, thresholdOpt)

    y_pred_train_smoothed = smooth_window(y_pred_train.copy(), 3)# change the time window 3 according to your study

    train_metrics = measures(y_train, y_pred_train_smoothed) + ["train"]
    test_metrics = measures(y_test, y_pred_test) + ["test"]

    results = pd.DataFrame([train_metrics, test_metrics], columns=[
        "Sensitivity", "Specificity", "Balanced Accuracy", "F1 Score", "PPV", "NPV", "Dataset"
    ])

    return results

# Use the functions below for hyperparameter tuning and cross-validation if needed
#We assume that each id/participant will go in a separate fold
def create_cv_sets(train_set, k):
    """
    Create cross-validation sets.

    Args:
        train_set (pd.DataFrame): Training data.
        k (int): Number of folds.

    Returns:
        list: Cross-validation folds.
    """
    unique_ids = train_set['id'].unique()
    np.random.shuffle(unique_ids)
    ids_split = np.array_split(unique_ids, k)

    cv_sets = [train_set[train_set['id'].isin(ids)] for ids in ids_split]
    fold = []
    for i in range(k):
        r = []
        r.append(np.array(cv_sets[i].drop(columns=['id']).values))
        r.append(np.array(cv_sets[i]['wear'].values).reshape(-1, 1))
        fold.append(r)
    return fold

def objective(params):
    """
    Objective function for hyperparameter tuning.

    Args:
        params (dict): Model parameters.

    Returns:
        dict: Loss and status.
    """
    k = 5
    folds = create_cv_sets(train_set, k)

    scores = []
    for i in range(k):
        X_val, y_val = folds[i]
        X_train_fold = np.concatenate([f[0] for j, f in enumerate(folds) if j != i])
        y_train_fold = np.concatenate([f[1] for j, f in enumerate(folds) if j != i])

        model = xgb.XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold)

        y_prob = model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_prob)
        youdenJ = tpr - fpr
        thresholdOpt = thresholds[np.argmax(youdenJ)]

        y_pred = binary_array(y_prob, thresholdOpt)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        scores.append(bal_acc)

    return {'loss': -np.mean(scores), 'status': STATUS_OK}

def hyperparameter_optimization(X_train, y_train):
    """
    Perform hyperparameter optimization using Hyperopt.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.

    Returns:
        dict: Best parameters.
    """
    param_space = {
        'tree_method': 'hist',
        'device': 'cuda',
        'sampling_method': 'uniform',
        'max_bin': hp.choice('max_bin', range(10, 500)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1.0),
        'max_depth': hp.choice('max_depth', range(1, 40)),
        'reg_alpha': hp.quniform('reg_alpha', 0, 10, 0.01),
        'n_estimators': hp.choice('n_estimators', range(50, 400)),
        'gamma': hp.uniform('gamma', 0, 15),
        'eta': hp.quniform('eta', 0.01, 1.0, 0.01),
        'reg_lambda': hp.quniform('reg_lambda', 0, 10, 0.01),
        'booster': 'gbtree',
        'rate_drop': hp.quniform('rate_drop', 0, 1, 0.01),
        'normalize_type': 'forest',
        'scale_pos_weight': hp.uniform('scale_pos_weight', 0.05, 1.0)
    }

    trials = Trials()
    best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50, trials=trials)
    return best_params

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost classifier.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.

    Returns:
        xgb.XGBClassifier: Trained XGBoost model.
    """
    model = xgb.XGBClassifier(
        tree_method='hist',
        booster='gbtree',
        normalize_type='forest',
        device='cuda',
        sampling_method='uniform',
        colsample_bytree=0.26742276705788653,
        gamma=12.76669619260072,
        learning_rate=0.008276044253310374,
        max_bin=489,
        max_depth=19,
        n_estimators=91,
        rate_drop=0.14,
        reg_alpha=5.44,
        reg_lambda=1.51,
        scale_pos_weight=0.15711409366745727
    )
    
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Path to the input CSV file
    file_path = input("Enter the path to the input CSV file: ")
    label_column = input("Enter the name of the binary label column: ")

    # Load and preprocess data
    df = load_and_preprocess_csv(file_path, label_column)

    # Split data
    X_train, y_train, X_test, y_test = split_data(df, label_column)

    # Train model
    model = train_xgboost(X_train, y_train)

    # Evaluate predictions
    metrics = evaluate_predictions(model, X_train, y_train, X_test, y_test)

    print(metrics)

    # Uncomment below for hyperparameter optimization for now we have put the best parameters found in our study
    #Below is the sample code to run the cross-validation and Hyperopt
    # best_params = hyperparameter_optimization(X_train, y_train)
    # print(f"Best Parameters: {best_params}")
