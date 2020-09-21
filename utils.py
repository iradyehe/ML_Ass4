import numpy as np
import pandas as pd
import os

from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def get_all_datasets_full_path():
    return get_all_files_path_in_folder('datasets')


def get_all_files_path_in_folder(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]


def get_filename(path):
    head, tail = head, tail = os.path.split(path)
    return os.path.splitext(tail)[0]


def create_csv_output_file():
    path = 'output.csv'
    headers = 'Dataset Name,Algorithm Name,Cross Validation [1 - 10],Hyper-Parameters Values,Accuracy,TPR,FPR, ' \
              'Precision,AUC,PR-Curve,Training Time,Inference Time\n'

    with open(path, "w") as file:
        file.write(headers)

    return path


def append_to_csv_file(path, line):
    with open(path, 'a') as file:
        file.write(line + '\n')


def split_datasets_to_x_y(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def preprocess_data(ds):
    # read the ds into df
    df = pd.read_csv(ds)

    # fill nulls
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # encoding categorical columns and label
    df = columns_and_label_encoder(df)

    # encoding categorical columns and label
    df = columns_and_label_encoder(df)

    # split to X, y
    X, y = split_datasets_to_x_y(df)

    return X, y


def columns_and_label_encoder(df, y_column=-1):
    X = df.drop(df.columns[[-1]], axis=1)
    X = pd.get_dummies(X)

    y = df[df.columns[[-1]]]
    y = y.apply(LabelEncoder().fit_transform)

    res = pd.concat([X, y], axis=1)
    res.reset_index(inplace=True, drop=True)

    return res


def split_to_train_test(X, y, test_size=0.3, random_state=123):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_classes_names(y):
    classes = np.unique(y)
    return classes


def dict_to_str(my_dict):
    return ''.join(f'{key}: {str(val)} | ' for key, val in my_dict.items())