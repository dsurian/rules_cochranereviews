# -*- coding: utf-8 -*-
from __future__ import division
import sys
import os
from os import listdir
import time
import csv
import datetime
import operator
import cPickle as pickle
import numpy as np
import sklearn
import io
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import ks_2samp
from sklearn.utils import resample
import scipy.stats
from scipy import stats
import math
import random
import warnings
warnings.filterwarnings('ignore')



def run_gridsearchcv_RFClassifier(X, y):
    parameters={'n_estimators': range(5,105,5),
                'criterion':['entropy','gini'],
                'class_weight':['balanced'],
                'max_features':['auto', 'sqrt', 'log2'],
                'max_depth': [2, 3, 4],
                'random_state':[42]
                }

    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, refit=True, iid=True)
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()
    optimised_clf = grid_search.best_estimator_
    optimised_params = grid_search.best_params_

    return optimised_clf, optimised_params

def run_gridsearchcv_DTClassifier(X, y):
    parameters = {"criterion": ["entropy", "gini"],
                  'class_weight': ['balanced'],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [2, 3, 4],
                  'random_state':[42]
                  }

    clf = tree.DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, refit=True, iid=True)
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()
    optimised_clf = grid_search.best_estimator_
    optimised_params = grid_search.best_params_

    return optimised_clf, optimised_params

def run_gridsearchcv_LogisticRegression(X, y):
    parameters = {'penalty': ['l1', 'l2'],
                  'class_weight': ['balanced'],
                  'solver': ['liblinear'],
                  'n_jobs': [-1],
                  'random_state':[42]
                  }

    clf = LogisticRegression()
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=10, refit=True, iid=True)
    grid_search.fit(X, y)
    best_parameters = grid_search.best_estimator_.get_params()
    optimised_clf = grid_search.best_estimator_
    optimised_params = grid_search.best_params_

    return optimised_clf, optimised_params


def get_stat(d_feature_d_conclusion_l_vals):
    for feature in d_feature_d_conclusion_l_vals:
        print ('{0}'.format(feature))

        d_conclusion_l_vals = d_feature_d_conclusion_l_vals[feature]

        l_lvals = []
        l_labels = []
        for conclusion in d_conclusion_l_vals:
            print ('\tConclusion: {0}'.format(conclusion))

            l_vals = d_conclusion_l_vals[conclusion]
            l_vals.sort()
            a_vals = np.asarray(l_vals)

            Q1 = np.percentile(a_vals, 25)
            Q2 = np.percentile(a_vals, 50)
            Q3 = np.percentile(a_vals, 75)
            IQR = Q3 - Q1

            l_lvals.append(l_vals)
            l_labels.append(conclusion)

            print ('\t\tQ1: {0}, Q2: {1}, Q3: {2}, IQR: {3}'.format(Q1, Q2, Q3, IQR))

        _, pval = ks_2samp(np.asarray(l_lvals[0]), np.asarray(l_lvals[1]))

        print ('\tp-val: {0}\n'.format(pval))

def confintv(scores):
    median = np.median(scores)

    alpha = 5.0
    lower_p = alpha / 2.0
    lower = max(0.0, np.percentile(scores, lower_p))
    upper_p = (100 - alpha) + (alpha / 2.0)
    upper = min(1.0, np.percentile(scores, upper_p))

    return median, lower, upper

def bootstrap(X, y, clf, acc_calc, prec_calc, recall_calc, fscore_calc, aucroc_calc, n=1000):
    random.seed(42)

    a_X = np.asarray(X)
    a_y = np.asarray(y)

    l_accuracy = []
    l_aucroc = []
    l_precision = []
    l_recall = []
    l_f1score = []

    for _ in xrange(n):
        l_ix_rand = [random.randint(0, X.shape[0]-1) for _ in range(X.shape[0])]
        a_data_rnd = a_X[l_ix_rand, :]
        a_labels_rnd = a_y[l_ix_rand]

        a_labels_pred = clf.predict(a_data_rnd)
        a_scores_pred = clf.predict_proba(a_data_rnd)[:, 1]

        acc = accuracy_score(a_labels_rnd, a_labels_pred)
        l_accuracy.append(acc)

        aucroc = roc_auc_score(a_labels_rnd, a_scores_pred)
        l_aucroc.append(aucroc)

        prec, recall, fscore, support = \
            precision_recall_fscore_support(a_labels_rnd, a_labels_pred, average='macro')

        l_precision.append(prec)
        l_recall.append(recall)
        l_f1score.append(fscore)

    # -- Confidence intervals
    alpha = 95
    print ('\tConfidence interval: {0}'.format(alpha))

    # -- Accuracy
    _, lower, upper = confintv(np.asarray(l_accuracy))
    print ('\t\tAccuracy: {2} ({0} - {1})'.format(lower * 100, upper * 100, acc_calc * 100))

    # -- Precision
    _, lower, upper = confintv(np.asarray(l_precision))
    print ('\t\tPrecision: {2} ({0} - {1})'.format(lower * 100, upper * 100, prec_calc * 100))

    # -- Recall
    _, lower, upper = confintv(np.asarray(l_recall))
    print ('\t\tRecall: {2} ({0} - {1})'.format(lower * 100, upper * 100, recall_calc * 100))

    # -- F1-score
    _, lower, upper = confintv(np.asarray(l_f1score))
    print ('\t\tF1-score: {2} ({0} - {1})'.format(lower * 100, upper * 100, fscore_calc * 100))

    # -- AUCROC
    _, lower, upper = confintv(np.asarray(l_aucroc))
    print ('\t\tAUCROC: {2} ({0} - {1})'.format(lower, upper, aucroc_calc))

def evaluate_test(clf, a_labels_test, X_test):
    a_labels_pred = clf.predict(X_test)
    a_scores_pred = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(a_labels_test, a_labels_pred)
    aucroc = roc_auc_score(a_labels_test, a_scores_pred)

    prec, recall, fscore, _ = \
        precision_recall_fscore_support(a_labels_test, a_labels_pred, average='macro')

    return acc, prec, recall, fscore, aucroc


def calc_num_days(date1, date2):
    return (date2 - date1).days

def readInput(inFile):
    f = open(inFile, 'rb')

    cntRow = 0
    data = []
    labels = []
    features_name = ['Number of trials in review', 'Number of participants in review', 'Coverage score', 'Update time using search dates']

    labels_name = ['Not changed', 'Changed']
    labels_name = np.asarray(labels_name, dtype='|S10')

    d_feature_d_conclusion_l_vals = {}

    for row in f:
        if cntRow == 0:
            cntRow += 1
            continue

        row = row.strip()
        col = row.split('\t')

        doi = col[0].strip()
        numTrials_2 = int(col[1].strip())
        numParticipants_2 = int(col[2].strip())
        searchDate_2 = datetime.datetime.strptime(col[3].strip(), '%d %B %Y')
        numTrials_3 = int(col[4].strip())
        numParticipants_3 = int(col[5].strip())
        searchDate_3 = datetime.datetime.strptime(col[6].strip(), '%d %B %Y')
        conclusion = int(col[7].strip())

        Update_time_using_search_dates = int(calc_num_days(searchDate_2, searchDate_3))

        if Update_time_using_search_dates < 0:
            print ('Negative Update_time. doi: {0}. searchDate_2: {1}. searchDate_3: {2}'.format(doi, searchDate_2, searchDate_3))
            sys.exit()

        Coverage_score = numParticipants_2 / numParticipants_3


        data.append([numTrials_2, numParticipants_2, Coverage_score,
                     Update_time_using_search_dates])
        labels.append(conclusion)

        # -- For statistics
        feature = 'Numb of trials (review)'
        if feature in d_feature_d_conclusion_l_vals:
            if conclusion in d_feature_d_conclusion_l_vals[feature]:
                d_feature_d_conclusion_l_vals[feature][conclusion].append(numTrials_2)
            else:
                d_feature_d_conclusion_l_vals[feature][conclusion] = [numTrials_2]
        else:
            d_feature_d_conclusion_l_vals[feature] = {conclusion: [numTrials_2]}

        feature = 'Numb of participants (review)'
        if feature in d_feature_d_conclusion_l_vals:
            if conclusion in d_feature_d_conclusion_l_vals[feature]:
                d_feature_d_conclusion_l_vals[feature][conclusion].append(numParticipants_2)
            else:
                d_feature_d_conclusion_l_vals[feature][conclusion] = [numParticipants_2]
        else:
            d_feature_d_conclusion_l_vals[feature] = {conclusion: [numParticipants_2]}

        feature = 'Coverage score'
        if feature in d_feature_d_conclusion_l_vals:
            if conclusion in d_feature_d_conclusion_l_vals[feature]:
                d_feature_d_conclusion_l_vals[feature][conclusion].append(Coverage_score)
            else:
                d_feature_d_conclusion_l_vals[feature][conclusion] = [Coverage_score]
        else:
            d_feature_d_conclusion_l_vals[feature] = {conclusion: [Coverage_score]}

        feature = 'Elapsed time between search dates'
        if feature in d_feature_d_conclusion_l_vals:
            if conclusion in d_feature_d_conclusion_l_vals[feature]:
                d_feature_d_conclusion_l_vals[feature][conclusion].append(Update_time_using_search_dates)
            else:
                d_feature_d_conclusion_l_vals[feature][conclusion] = [Update_time_using_search_dates]
        else:
            d_feature_d_conclusion_l_vals[feature] = {conclusion: [Update_time_using_search_dates]}


    data = np.asarray(data)
    labels = np.asarray(labels)

    return data, labels, features_name, labels_name, d_feature_d_conclusion_l_vals



def check_exist_dir(dirname):
    return os.path.isdir(dirname)

def check_exist(fname):
    return os.path.isfile(fname)

def dump_var(data, outFile):
    pickle.dump(data, open(outFile + '.cpickle', "wb"), protocol=2)

def load_var(inFile):
    return pickle.load(open(inFile + '.cpickle', "rb"))

def write_to_file(outFile, text, mode):
    with open(outFile, mode) as oF:
        oF.write(text)

def done():
    print ('\nFinish')
    sys.exit()


if __name__ == '__main__':

    resultFolder = 'Results/'
    cpickleFolder = 'cpickle/'


    LOAD_PREVIOUS = False
    RETRAINED = False

    print ('[0] Load previous trained classifiers')
    print ('[1] Retrained the classifiers on your dataset')
    ipt = input("Option: ")

    if ipt == 0: LOAD_PREVIOUS = True
    elif ipt == 1: RETRAINED = True
    else:
        print ('Please only input 0 or 1')
        sys.exit()

    if LOAD_PREVIOUS:
        print ('> Load previous trained classifiers ...')

        # -- Read data
        inFile = resultFolder + 'extracted_info.txt'
        X, y, features_name, labels_name, d_feature_d_conclusion_l_vals = readInput(inFile)
        l_y = list(y)

        print ('\n')
        get_stat(d_feature_d_conclusion_l_vals)
        print ('\n')

        # -- Training/Test sets
        # ----------------------
        train_size = 0.8
        test_size = 1 - train_size
        l_ix = range(len(y))
        X_train, X_test, y_train, y_test, ix_train, ix_test = \
            train_test_split(X, y, l_ix, test_size=test_size, stratify=l_y, random_state=30)

        a_labels_test = np.asarray(y_test)

        # -- Classifiers
        # ----------------------
        print ('\nLogistic Regression')
        print ('------------')
        optimised_LR = load_var(resultFolder + cpickleFolder + 'optimised_LR')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_LR, a_labels_test, X_test)

        print ('Bootstrap:')
        bootstrap(X_test, a_labels_test, optimised_LR, acc, prec, recall, fscore, aucroc)

        print ('Feature importance:')
        featimportance = optimised_LR.coef_[0]
        top = np.argpartition(featimportance, -4)[-4:]
        top_sorted = top[np.argsort(featimportance[top])]
        for i in top_sorted:
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i]))


        print ('\nDecision Tree')
        print ('------------')
        optimised_DT = load_var(resultFolder + cpickleFolder + 'optimised_DT')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_DT, a_labels_test, X_test)

        print ('Bootstrap:')
        bootstrap(X_test, a_labels_test, optimised_DT, acc, prec, recall, fscore, aucroc)

        print ('Feature importance:')
        featimportance = optimised_DT.feature_importances_
        for i in xrange(len(featimportance)):
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i]))


        print ('\nRandom Forest')
        print ('------------')
        optimised_RF = load_var(resultFolder + cpickleFolder + 'optimised_RF')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_RF, a_labels_test, X_test)

        print ('Bootstrap:')
        bootstrap(X_test, a_labels_test, optimised_RF, acc, prec, recall, fscore, aucroc)

        print ('Feature importance:')
        featimportance = optimised_RF.feature_importances_
        for i in xrange(len(featimportance)):
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i]))

        done()

    if RETRAINED:
        print ('> Retrained the classifiers on your dataset ...')

        yourFolder = ''
        while yourFolder.strip() == '':
            yourFolder = raw_input("> Enter your folder name: ")

        resultFolder = yourFolder + '/' + resultFolder

        if not check_exist_dir(resultFolder):
            print ('{0} does not exist. Please refer to README file.'.format(resultFolder))
            sys.exit()


        # -- Read data
        inFile = resultFolder + 'extracted_info.txt'
        X, y, features_name, labels_name, d_feature_d_conclusion_l_vals = readInput(inFile)
        l_y = list(y)

        print ('\n')
        get_stat(d_feature_d_conclusion_l_vals)
        print ('\n')

        # -- Training/Test sets
        # ----------------------
        train_size = 0.8
        test_size = 1 - train_size
        l_ix = range(len(y))
        X_train, X_test, y_train, y_test, ix_train, ix_test = \
            train_test_split(X, y, l_ix, test_size=test_size, stratify=l_y, random_state=30)

        a_labels_test = np.asarray(y_test)

        # -- Classifiers
        # ----------------------
        print ('\nLogistic Regression')
        print ('------------')
        optimised_LR, _ = run_gridsearchcv_LogisticRegression(X_train, y_train)
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_LR, a_labels_test, X_test)

        print ('Bootstrap:')
        bootstrap(X_test, a_labels_test, optimised_LR, acc, prec, recall, fscore, aucroc)

        print ('Feature importance:')
        featimportance = optimised_LR.coef_[0]
        top = np.argpartition(featimportance, -4)[-4:]
        top_sorted = top[np.argsort(featimportance[top])]
        for i in top_sorted:
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i]))


        print ('\nDecision Tree')
        print ('------------')
        optimised_DT, _ = run_gridsearchcv_DTClassifier(X_train, y_train)
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_DT, a_labels_test, X_test)

        print ('Bootstrap:')
        bootstrap(X_test, a_labels_test, optimised_DT, acc, prec, recall, fscore, aucroc)

        print ('Feature importance:')
        featimportance = optimised_DT.feature_importances_
        for i in xrange(len(featimportance)):
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i]))


        print ('\nRandom Forest')
        print ('------------')
        optimised_RF, _ = run_gridsearchcv_RFClassifier(X_train, y_train)
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_RF, a_labels_test, X_test)

        print ('Bootstrap:')
        bootstrap(X_test, a_labels_test, optimised_RF, acc, prec, recall, fscore, aucroc)

        print ('Feature importance:')
        featimportance = optimised_RF.feature_importances_
        for i in xrange(len(featimportance)):
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i]))

        done()

