import numpy as np
import pandas as pd

from sklearn.ensemble.forest import _generate_unsampled_indices, _get_n_samples_bootstrap
from sklearn.ensemble import RandomForestClassifier


class BROOF:

    def __init__(self, M, n_trees, max_depth):

        self.M = M
        self.n_trees = n_trees
        self.max_depth = max_depth

        self.alphas = []
        self.models = []

    # Line 1:
    def fit(self, X_train, y_train):

        # Initialize the weights of each sample with wi = 1/N and create a dataframe in which the evaluation is computed
        Evaluation = pd.DataFrame(y_train.copy())
        Evaluation.rename(columns={0: 'target'}, inplace=True)

        # Line 2:
        Evaluation['weights'] = 1 / y_train.shape[0]  # Set the initial weights w = 1/N

        # Line 3:
        for t in range(self.M):
            # Line 4:
            model, unsampled_indices = self.learn_rf(X_train, y_train, t)

            # Append the single weak classifiers to a list which is later on used to make the weighted decision
            self.models.append(model)
            predictions = model.predict(X_train)

            # Add values to the Evaluation DataFrame
            Evaluation['predictions'] = predictions
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation['target'], 1, 0)
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation['target'], 1, 0)

            oob_samples_misclassified = Evaluation[Evaluation.index.isin(unsampled_indices)]['misclassified']
            oob_samples_weights = Evaluation[Evaluation.index.isin(unsampled_indices)]['weights']

            # Line 5:
            numerator = np.sum(oob_samples_misclassified * oob_samples_weights)
            denominator = np.sum(Evaluation['weights'])
            oob_w_err = numerator / denominator

            # Line 6:
            if oob_w_err == 0:
                alpha_m = 0
            else:
                alpha_m = np.log((1 - oob_w_err) / oob_w_err)

            self.alphas.append(alpha_m)

            # Line 7:
            oob_samples_weights *= np.exp(alpha_m * oob_samples_misclassified)

            # update the weights according the oob_samples
            for items in oob_samples_weights.iteritems():
                Evaluation.at[items[0], 'weights'] = items[1]

    def learn_rf(self, X_train, y_train, t):
        rf_model = RandomForestClassifier(n_estimators=self.n_trees, max_depth=self.max_depth, bootstrap=True, random_state=t, verbose=0)

        model = rf_model.fit(X_train, y_train)

        n_samples = X_train.shape[0]
        num_estimators = len(model.estimators_)
        unsampled_indices = _generate_unsampled_indices(
            model.estimators_[0].random_state, n_samples,
            _get_n_samples_bootstrap(n_samples, None))

        for ind in range(1, num_estimators):
            arr = _generate_unsampled_indices(
                model.estimators_[ind].random_state, n_samples,
                _get_n_samples_bootstrap(n_samples, None))
            unsampled_indices = np.unique(np.concatenate((unsampled_indices, arr), 0))

        return model, unsampled_indices

    def predict_proba(self, X_test):
        predictions = []
        sum_alphas = np.sum(self.alphas, axis=0)
        if sum_alphas == 0:
            for alpha, model in zip(self.alphas, self.models):
                pred_poba = model.predict_proba(X_test)
                predictions.append(pred_poba)
            sum_pred = np.sum(np.array(predictions), axis=0)
            temp_calc = sum_pred / len(self.alphas)
        else:
            for alpha, model in zip(self.alphas, self.models):
                pred_poba = model.predict_proba(X_test)
                y_pred = alpha * pred_poba
                predictions.append(y_pred)
            sum_pred = np.sum(np.array(predictions), axis=0)
            temp_calc = sum_pred / sum_alphas
        return temp_calc

    def predict(self, X_test):
        predictions = []
        sum_alphas = np.sum(self.alphas, axis=0)
        if sum_alphas == 0:
            for alpha, model in zip(self.alphas, self.models):
                pred_poba = model.predict_proba(X_test)
                predictions.append(pred_poba)
            sum_pred = np.sum(np.array(predictions), axis=0)
            temp_calc = sum_pred / len(self.alphas)
            arg_maxes = np.argmax(temp_calc, axis=1)
        else:
            for alpha, model in zip(self.alphas, self.models):
                pred_poba = model.predict_proba(X_test)
                y_pred = alpha * pred_poba
                predictions.append(y_pred)
            sum_pred = np.sum(np.array(predictions), axis=0)
            temp_calc = sum_pred / sum_alphas
            arg_maxes = np.argmax(temp_calc, axis=1)

        return arg_maxes

    def get_params(self, deep=True):
        out = {
            'M': self.M,
            'n_trees': self.n_trees,
            'max_depth': self.max_depth
        }

        return out

    def set_params(self, **params):
        self.n_trees = params['n_trees']
        self.M = params['M']
        return self