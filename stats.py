import numpy as np
import pandas as pd
import os


def friedman_test(path):
    print('friedman test on file {}'.format(path))
    out_df = pd.read_csv(path)
    accuracy_df = calc_avg_accuracy(out_df)
    ranks_df = calc_ranks(accuracy_df)
    R_j_broof, R_j_xgb = calc_R_j(ranks_df)
    X_2_f = calc_X_2_f(len(ranks_df['dataset'].values), 2, R_j_broof, R_j_xgb)
    F_F = calc_F_F(len(ranks_df['dataset'].values), 2, X_2_f)
    print(accuracy_df.head(4))
    print(ranks_df.head(4))
    print('{}, {}'.format(R_j_broof, R_j_xgb))
    print(X_2_f)
    print(F_F)

    # if F_F > the cell at F(1, inf) at the table from:
    # https://home.ubalt.edu/ntsbarsh/Business-stat/StatistialTables.pdf
    # then it is statistically significant. in our case it isnt (for now - using 127 datasets)
    # TODO: run it again once we could get data on all datasets


def calc_F_F(N, L, X_2_f):
    return ( (N-1) * X_2_f ) / ( N*(L-1) - X_2_f )


def calc_X_2_f(N, L, R_j_broof, R_j_xgb):
    return ( (12*N) / (L*(L+1)) ) * ( ( R_j_broof**2 + R_j_xgb**2 ) - ( ( L*((L+1)**2) ) / 4 ) )


def calc_R_j(df):

    R_j_broof = sum(df['rank_broof'].values)
    R_j_xgb = sum(df['rank_xgb'].values)

    R_j_broof /= len(df['rank_broof'].values)
    R_j_xgb /= len(df['rank_xgb'].values)

    return R_j_broof, R_j_xgb



def calc_ranks(df):
    datasets = []
    ranks_broof = []
    ranks_xgb = []

    ranks_df = pd.DataFrame()

    for i, row in df.iterrows():
        datasets.append(row['dataset'])
        if row['accuracy_broof'] > row['accuracy_xgb']:
            ranks_broof.append(1)
            ranks_xgb.append(2)
        else:
            ranks_broof.append(2)
            ranks_xgb.append(1)

    ranks_df['dataset'] = datasets
    ranks_df['rank_broof'] = ranks_broof
    ranks_df['rank_xgb'] = ranks_xgb

    return ranks_df


def calc_avg_accuracy(df):
    datasets = []
    accuracies_broof = []
    accuracies_xgb = []

    avg_accs = pd.DataFrame()

    temp_acc_broof = []
    temp_acc_xgb = []
    for i, row in df.iterrows():
        if row['Algorithm Name'] == ' BROOF':
            temp_acc_broof.append(row['Accuracy'])
        else:
            temp_acc_xgb.append(row['Accuracy'])

        if len(temp_acc_broof) == 10 and len(temp_acc_xgb) == 10:
            aab = sum(temp_acc_broof) / 10
            aax = sum(temp_acc_xgb) / 10
            datasets.append(row['Dataset Name'])
            accuracies_broof.append(aab)
            accuracies_xgb.append(aax)
            temp_acc_broof = []
            temp_acc_xgb = []

    avg_accs['dataset'] = datasets
    avg_accs['accuracy_broof'] = accuracies_broof
    avg_accs['accuracy_xgb'] = accuracies_xgb

    return avg_accs


if __name__ == '__main__':
    friedman_test('output.csv')