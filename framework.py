import utils
from broof import BROOF

import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


class Framework():

    def __init__(self):
        self.__datasets = utils.get_all_datasets_full_path()
        self.counter = 0
        self.model_wins = 0

    def start(self):
        for ds in self.__datasets:
            self.counter += 1
            print(f'\n\nDataset: {ds}')
            X, y = utils.preprocess_data(ds)
            n_classes = utils.get_num_of_classes(y)
            X_train, X_test, y_train, y_test = utils.split_to_train_test(X, y)
            self.fit_and_predict(X_train, X_test, y_train, y_test, n_classes)

        print(f'model won in {self.model_wins} \ {self.counter}')



    def fit_and_predict(self, X_train, X_test, y_train, y_test, n_classes):
        print(self.counter)
        model = BROOF(M=10, n_trees=5)
        cv_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)
        print(cv_scores)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # XGBoost classifier
        xgb_acc = self.xgb_classifier(X_train, y_train, X_test, y_test, n_classes)
        print(f"\n----- model: {accuracy} , xgb: {xgb_acc} -----\n")

        if accuracy >= xgb_acc:
            self.model_wins += 1
            winner = 'model'
        else:
            winner = 'xgb'

        print(f'Winner: {winner}')


    def xgb_classifier(self, X_train, y_train, X_test, y_test, n_classes):
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        return accuracy_score(y_test, y_pred)


if __name__ == '__main__':
    frm = Framework()
    frm.start()