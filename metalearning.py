import pandas as pd

import random

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot

def metalearning(classificationfeatures):
    metafeatures = pd.read_csv(classificationfeatures)
    print(metafeatures.head())
    metafeatures_labeled = create_dummy_labels(metafeatures)  # TODO: after the framework is complete, use create_labels() function
    print(metafeatures_labeled.head())

    for i, row in metafeatures_labeled.iterrows():
        leave_one_out_model(metafeatures_labeled, i)


def create_dummy_labels(df):
    labels = []
    for i, row in df.iterrows():
        label = random.uniform(0, 1)
        labels.append(1 if label > 0.5 else 0)

    df['label'] = labels
    return df


def create_labels(df):
    pass


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

    plot_importance(model)
    pyplot.savefig("feature importances/" + test_name + "_feature_importances.png")
    # pyplot.show()

    # TODO: cross validation wrapper with k = n --> meaning leave one out
    # TODO: plot just one feature importance

    print(test_name)
    # print(i)
    # print(test)
    # print(train.head())

if __name__ == '__main__':
    metalearning('ClassificationAllMetaFeatures.csv')