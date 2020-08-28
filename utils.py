import pandas as pd
from os import listdir
from os.path import isfile, join

from sklearn.model_selection import train_test_split


def get_all_datasets_full_path():
    return get_all_files_path_in_folder('datasets')


def get_all_files_path_in_folder(folder_path):
    return [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]


def split_datasets_to_x_y(dataset_path):
    data = pd.read_csv(dataset_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def split_to_train_test(X, y, test_size=0.3, random_state=123):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
