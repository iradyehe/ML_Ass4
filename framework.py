import utils
import time
from broof import BROOF

import xgboost as xgb

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


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
            y = label_binarize(y, classes=self.classes_names)
            self.k_folds_cross_validation(X, y)

        print(f'model won in {self.model_wins} \ {self.counter}')

    def k_folds_cross_validation(self, X, y):
        kf = KFold(n_splits=10, shuffle=False)
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
        model = BROOF(M=10, n_trees=5, max_depth=10)
        model_to_set = OneVsRestClassifier(model)

        param_dist = {
            'estimator__M': [2, 5, 8, 10],
            'estimator__n_trees': [2, 5, 10, 15],
            'estimator__max_depth': [10, 50, 100, None]
        }

        model_tuning = self.randomized_search_creation(model_to_set, param_dist)
        return model_tuning

    def xgb_classifier(self):
        model = xgb.XGBClassifier(random_state=42)

        model_to_set = OneVsRestClassifier(model)

        param_dist = {
            'estimator__max_depth': range(3, 10, 2),
            'estimator__min_child_weight': range(1, 6, 2),
            'estimator__eta': [.3, .2, .1, .05, .01, .005]
        }

        model_tuning = self.randomized_search_creation(model_to_set, param_dist)
        return model_tuning

    def predicting(self, model, X_test, y_test):
        res_dict = {}
        # predict (should measure the time)
        predict_start_time = time.time()
        y_pred = model.predict(X_test)
        curr_predict_time = time.time() - predict_start_time
        y_score = model.predict_proba(X_test)
        # predict time for 1000 samples
        predict_time = curr_predict_time * (1000 / X_test.shape[0])

        TPR = FPR = Precision = Accuracy = AUC = PR_Curve = 0
        # Compute ROC curve and ROC area for each class PER FOLD
        for i in range(self.num_of_classes):
            cm = metrics.confusion_matrix(y_test[:, i], y_pred[:, i])
            try:
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                TPR += tp / (tp + fn) if tp != 0 else (1 if fn == 0 else 0)
                FPR += fp / (fp + tn) if fp != 0 else (1 if tn == 0 else 0)
                Precision += tp / (tp + fp) if tp != 0 else (1 if fp == 0 else 0)
            except:
                a = 1


        TPR /= self.num_of_classes
        FPR /= self.num_of_classes
        Precision /= self.num_of_classes
        Accuracy = metrics.accuracy_score(y_test, y_pred)

        AUC = metrics.roc_auc_score(y_test, y_score, multi_class="ovr", average="micro")
        PR_Curve = metrics.average_precision_score(y_test, y_score, average="micro")

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