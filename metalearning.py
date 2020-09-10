import pandas as pd

import random

import xgboost as xgb

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
    test = df.iloc[i]
    train = df.drop(df.index[i])

    model = xgb.XGBClassifier(random_state=42, objective='binary:logistic')

    print(test['dataset'])
    # print(i)
    # print(test)
    # print(train.head())


def write_result_table_to_file(self, algo_name, best_params_str, value_dict, train_time):
    res = f"{self.current_dataset_name}, {algo_name}, {self.cv_iteration_number}, {best_params_str}, " \
          f"{value_dict['Accuracy']:.2f}, {value_dict['TPR']:.2f}, {value_dict['FPR']:.2f}, " \
          f"{value_dict['Precision']:.2f}, {value_dict['AUC']:.2f}, {value_dict['PR_Curve']:.2f}, {train_time:.2f}," \
          f" {value_dict['InferenceTime']:.2f}"

if __name__ == '__main__':
    metalearning('ClassificationAllMetaFeatures.csv')