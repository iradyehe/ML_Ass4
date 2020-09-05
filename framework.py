import utils
import time
from broof import BROOF

import xgboost as xgb

from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold


class Framework():

    def __init__(self):
        self.__datasets = utils.get_all_datasets_full_path()
        self.counter = 0
        self.model_wins = 0
        self.output_csv_file_path = utils.create_csv_output_file()

    def start(self):
        for ds in self.__datasets:
            print(ds)
            self.current_dataset_name = utils.get_filename(ds)
            X, y = utils.preprocess_data(ds)
            self.classes_names = utils.get_classes_names(y)
            self.num_of_classes = len(self.classes_names)
            self.k_folds_cross_validation(X, y)

        print(f'model won in {self.model_wins} \ {self.counter}')

    def k_folds_cross_validation(self, X, y):
        kf = StratifiedKFold(n_splits=10, shuffle=False)
        self.cv_iteration_number = 1
        for train_index, test_index in kf.split(X, y):
            # Split train-test
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model
            self.fit_and_predict(X_train, X_test, y_train, y_test)

            self.cv_iteration_number += 1

    def fit_and_predict(self, X_train, X_test, y_train, y_test):
        # BROOF model training & testing
        broof_cv = self.broof_classiefier()
        best_broof, best_params, train_time = self.randomized_search_fit(broof_cv, X_train, y_train)
        best_params_str = utils.dict_to_str(best_params)
        value_dict = self.predicting(best_broof, X_test, y_test)
        self.write_result_table_to_file('BROOF', best_params_str, value_dict, train_time)

        # XGBoost model training & testing
        xgb_cv = self.xgb_classifier()
        best_xgb, best_params, train_time = self.randomized_search_fit(xgb_cv, X_train, y_train)
        value_dict = self.predicting(best_xgb, X_test, y_test)
        self.write_result_table_to_file('XGBoost', best_params_str, value_dict, train_time)

    def randomized_search_fit(self, model, X_train, y_train):
        print("Randomized search..")
        search_start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - search_start_time
        return model.best_estimator_, model.best_params_, train_time

    def randomized_search_creation(self, model, params):
        return RandomizedSearchCV(model, params, n_iter=12, scoring='accuracy', cv=3)

    def broof_classiefier(self):
        model = BROOF(M=10, n_trees=5)
        param_dist = {'M': [2, 5, 10, 15],
                      'n_trees': [5, 10, 15, 20]}

        model_cv = self.randomized_search_creation(model, param_dist)
        return model_cv

    def xgb_classifier(self):
        if self.num_of_classes > 2:
            model = xgb.XGBClassifier(random_state=42, objective='multi:softmax')
        else:
            model = xgb.XGBClassifier(random_state=42)
        param_dist = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
        }

        model_cv = self.randomized_search_creation(model, param_dist)
        return model_cv

    def predicting(self, model, X_test, y_test):
        res_dict = {}
        # predict (should measure the time)
        predict_start_time = time.time()
        y_pred = model.predict(X_test)
        curr_predict_time = time.time() - predict_start_time
        # predict time for 1000 samples
        predict_time = curr_predict_time * (1000 / X_test.shape[0])

        TPR = FPR = Precision = Accuracy = AUC = PR_Curve = 0
        if self.num_of_classes > 2:
            y_pred_prob = model.predict_proba(X_test)
            confusion_matrices = metrics.multilabel_confusion_matrix(y_test, y_pred)
            TPR = FPR = Precision = 0
            for cm in confusion_matrices:
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                TPR += tp / (tp + fn)
                FPR += fp / (fp + tn)
                Precision += tp / (tp + fp) if tp != 0 else (1 if fp == 0 else 0)

            TPR /= self.num_of_classes
            FPR /= self.num_of_classes
            Precision /= self.num_of_classes
            Accuracy = metrics.accuracy_score(y_test, y_pred)
            AUC = metrics.roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="macro")
            PR_Curve = 1#metrics.average_precision_score(y_test, y_pred, average="macro")
        else:
            cm = metrics.confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            TPR = tp / (tp + fn)
            FPR = fp / (fp + tn)
            Precision = tp / (tp + fp) if tp != 0 else (1 if fp == 0 else 0)
            Accuracy = metrics.accuracy_score(y_test, y_pred)
            AUC = metrics.roc_auc_score(y_test, y_pred)
            PR_Curve = metrics.average_precision_score(y_test, y_pred)

        res_dict['TPR'] = TPR
        res_dict['FPR'] = FPR
        res_dict['Precision'] = Precision
        res_dict['Accuracy'] = Accuracy
        res_dict['AUC'] = AUC
        res_dict['PR_Curve'] = PR_Curve
        res_dict['InferenceTime'] = predict_time

        return res_dict

    def write_result_table_to_file(self, algo_name, best_params_str, value_dict, train_time):
        res = f"{self.current_dataset_name}, {algo_name}, {self.cv_iteration_number}, {best_params_str}, " \
              f"{value_dict['Accuracy']:.2f}, {value_dict['TPR']:.2f}, {value_dict['FPR']:.2f}, " \
              f"{value_dict['Precision']:.2f}, {value_dict['AUC']:.2f}, {value_dict['PR_Curve']:.2f}, {train_time:.2f}," \
              f" {value_dict['InferenceTime']:.2f}"

        utils.append_to_csv_file(self.output_csv_file_path, res)


if __name__ == '__main__':
    frm = Framework()
    frm.start()