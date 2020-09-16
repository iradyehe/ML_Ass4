import pandas as pd
import numpy as np
import csv

import random

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot

from sklearn.model_selection import train_test_split

def metalearning(classificationfeatures):
    metafeatures = pd.read_csv(classificationfeatures)
    print(metafeatures.head())
    metafeatures_labeled = create_labels(metafeatures)  # TODO: after the framework is complete, use create_labels() function
    print(metafeatures_labeled.head())

    predicted_labels = []
    for i, row in metafeatures_labeled.iterrows():
        pred_i = leave_one_out_model(metafeatures_labeled, i)
        predicted_labels.append(pred_i)

    print(predicted_labels[:5])

    hit_rate = []
    # Iterating the index
    # same as 'for i in range(len(list))'
    for i in range(len(predicted_labels)):
        p = predicted_labels[i]
        a = metafeatures['label'].values[i]
        if p == a:
            hit_rate.append(1)
        else:
            hit_rate.append(0)

    print('accuracy: {}'.format(sum(hit_rate) / len(hit_rate)))

    feature_importances_calc(metafeatures_labeled)


def create_dummy_labels(df):
    labels = []
    for i, row in df.iterrows():
        label = random.uniform(0, 1)
        labels.append(1 if label > 0.5 else 0)

    df['label'] = labels
    return df


def create_labels(df):
    labels = []
    output_df = pd.read_csv('output.csv')
    for i in range(int(output_df.shape[0] / 20)):  # 0 - 149
        accuracy_acc_broof = 0
        accuracy_acc_xgb = 0
        for j in range(0, 20, 2):
            row_broof = output_df.iloc[[i*20 + j]]
            row_xgb = output_df.iloc[[i*20 + j + 1]]
            accuracy_acc_broof += row_broof.iloc[0]['Accuracy']
            accuracy_acc_xgb += row_xgb.iloc[0]['Accuracy']
        accuracy_acc_broof /= 10
        accuracy_acc_xgb /= 10
        labels.append(1 if accuracy_acc_xgb < accuracy_acc_broof else 0)

    # todo - remove once all the datasets are good, else it will not work
    t = len(labels)
    for i in range(df.shape[0] - t):
        labels.append(1)

    df['label'] = labels
    return df


def leave_one_out_model(df, i):

    test = df.iloc[[i]]
    test_name = test.iloc[0].loc['dataset']
    test = test.drop(['dataset'], axis=1)
    train = df.drop(df.index[i]).drop(['dataset'], axis=1)

    X_train = train.drop(['label'], axis=1)
    y_train = train['label']
    X_test = test.drop(['label'], axis=1)
    y_test = test['label']

    model = xgb.XGBClassifier(random_state=42, objective='binary:logistic')
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    # TODO: cross validation wrapper with k = n --> meaning leave one out

    print(test_name)
    # print(i)
    # print(test)
    # print(train.head())

    return pred[0]


def feature_importances_calc(df):
    dataset = df.drop(['dataset'], axis=1)
    X = dataset.drop(['label'], axis=1)
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = xgb.XGBClassifier(random_state=42, objective='binary:logistic')
    model.fit(X_train, y_train)

    plot_importance(model, importance_type='weight', max_num_features=10)
    pyplot.savefig("feature importances/feature_importances_weight.png")
    pyplot.close('all')
    plot_importance(model, importance_type='gain', max_num_features=10)
    pyplot.savefig("feature importances/feature_importances_gain.png")
    pyplot.close('all')
    plot_importance(model, importance_type='cover', max_num_features=10)
    pyplot.savefig("feature importances/feature_importances_cover.png")
    pyplot.close('all')

    importance_weight = model.get_booster().get_score(importance_type='weight')
    importance_gain = model.get_booster().get_score(importance_type='gain')
    importance_cover = model.get_booster().get_score(importance_type='cover')

    with open('feature importances/importance_weight.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, importance_weight.keys())
        w.writeheader()
        w.writerow(importance_weight)

    with open('feature importances/importance_gain.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, importance_gain.keys())
        w.writeheader()
        w.writerow(importance_gain)

    with open('feature importances/importance_cover.csv', 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, importance_cover.keys())
        w.writeheader()
        w.writerow(importance_cover)

    shap_values = model.get_booster().predict(xgb.DMatrix(X_test), pred_contribs=True)
    shap_values_df = pd.DataFrame(shap_values)
    shap_values_df.columns = dataset.columns
    shap_values_df.to_csv('feature importances/shap_values.csv')

    print('nigerundayooo')


if __name__ == '__main__':
    metalearning('ClassificationAllMetaFeatures.csv')