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
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu
from sklearn.utils import resample
import scipy.stats
from scipy import stats
import math
import random
import warnings
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from matplotlib.offsetbox import AnchoredText
from collections import Counter
warnings.filterwarnings('ignore')




def ci_bootstrap_calibrationcurve(clf, X, y, numbsampling=2000):
    np.random.seed(42)

    d_group_d_xy_l = {}
    for _ in xrange(numbsampling):
        l_ix_rand = np.random.choice(len(y), replace=True, size=len(y)).tolist()
        X_sample = X[l_ix_rand, :]
        y_sample = y[l_ix_rand]

        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_sample)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_sample)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_of_positives_ci_y, mean_predicted_value_ci_x = calibration_curve(y_sample, prob_pos, n_bins=10)

        l_ixsort = list(np.argsort(mean_predicted_value_ci_x))
        l_y = list(fraction_of_positives_ci_y)
        l_x = list(mean_predicted_value_ci_x)
        for i in xrange(len(l_ixsort)):
            ix = l_ixsort[i]

            ci_x = l_x[ix]
            ci_y = l_y[ix]
            if i in d_group_d_xy_l:
                d_group_d_xy_l[i]['x'].append(ci_x)
                d_group_d_xy_l[i]['y'].append(ci_y)
            else:
                d_group_d_xy_l[i] = {'x': [ci_x], 'y': [ci_y]}

    ltup_lower_upper_x = []
    ltup_lower_upper_y = []
    for i in xrange(len(d_group_d_xy_l)):
        l_x = d_group_d_xy_l[i]['x']
        l_y = d_group_d_xy_l[i]['y']

        lower_x, upper_x = confint(l_x)
        lower_y, upper_y = confint(l_y)
        ltup_lower_upper_x.append((lower_x, upper_x))
        ltup_lower_upper_y.append((lower_y, upper_y))

    return ltup_lower_upper_x, ltup_lower_upper_y

# ---
# Calibration curve
#   y: prob true; the proportion of samples whose class is the positive class, in each bin (fraction of positives)
#   x: prob pred; the mean predicted probability in each bin
def calibration(X_test, y_test, l_clf_modelname):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    d_name_l_mean_predicted_value_x = {}
    d_name_l_fraction_of_positives_y = {}
    d_name_ltup_lower_upper_x_ci = {}
    d_name_ltup_lower_upper_y_ci = {}
    d_name_brierscore = {}
    for clf, name in l_clf_modelname:
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        l_ixsort = list(np.argsort(mean_predicted_value))
        l_fraction_of_positives = list(fraction_of_positives)
        l_mean_predicted_value = list(mean_predicted_value)
        l_mean_predicted_value_x = []
        l_fraction_of_positives_y = []
        for i in xrange(len(l_ixsort)):
            ix = l_ixsort[i]
            l_mean_predicted_value_x.append(l_mean_predicted_value[ix])
            l_fraction_of_positives_y.append(l_fraction_of_positives[ix])
        d_name_l_mean_predicted_value_x[name] = l_mean_predicted_value_x
        d_name_l_fraction_of_positives_y[name] = l_fraction_of_positives_y

        ltup_lower_upper_x, ltup_lower_upper_y = ci_bootstrap_calibrationcurve(clf, X_test, y_test)
        d_name_ltup_lower_upper_x_ci[name] = ltup_lower_upper_x
        d_name_ltup_lower_upper_y_ci[name] = ltup_lower_upper_y

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_test.max())
        print("%s:" % name)
        print("\tBrier score: %1.3f" % (clf_score))
        d_name_brierscore[name] = clf_score

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-.",
                 label="%s" % (name,))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    stbox = 'Brier score:\nLogistic regression: {0}\nDecision tree: {1}\nRandom forest: {2}'.format(
        d_name_brierscore['Logistic regression'],
        d_name_brierscore['Decision tree'],
        d_name_brierscore['Random forest'])
    anchored_text = AnchoredText(stbox, loc=2)
    ax1.add_artist(anchored_text)

    plt.tight_layout()
    fname = 'calibration.pdf'
    plt.savefig(fname, bbox_inches='tight', format='pdf')
    plt.show()

    for name in d_name_l_mean_predicted_value_x:
        print ('{0}'.format(name))
        l_mean_predicted_value_x = d_name_l_mean_predicted_value_x[name]
        l_fraction_of_positives_y = d_name_l_fraction_of_positives_y[name]

        for ix in xrange(len(l_mean_predicted_value_x)):
            x = l_mean_predicted_value_x[ix]
            y = l_fraction_of_positives_y[ix]
            tup_lower_upper_ci_x = d_name_ltup_lower_upper_x_ci[name][ix]
            tup_lower_upper_ci_y = d_name_ltup_lower_upper_y_ci[name][ix]

            print ('\t{0} [{1}-{2}], {3} [{4}-{5}]'.format(x, tup_lower_upper_ci_x[0], tup_lower_upper_ci_x[1],
                                            y, tup_lower_upper_ci_y[0], tup_lower_upper_ci_y[1]))

def get_acc_aucroc_prec_recall_f1score(y_sample, y_sample_pred, scores_sample_pred):
    acc = accuracy_score(y_sample, y_sample_pred)
    aucroc = roc_auc_score(y_sample, scores_sample_pred)
    prec, recall, f1score, support = \
        precision_recall_fscore_support(y_sample, y_sample_pred, average='macro')

    return acc, aucroc, prec, recall, f1score

def confint(data):
    alpha = 0.05
    estimate = np.mean(data)
    lower = np.percentile(data, 100 * (alpha / 2))
    upper = np.percentile(data, 100 * (1 - alpha / 2))
    stderr = np.std(data)

    return lower, upper

def ci_bootstrap_train(clf_name, optimised_clf, optimised_clf_param, X, y, features_name,
                 acc_calc, prec_calc, recall_calc, fscore_calc, aucroc_calc, n=2000):
    np.random.seed(42)

    d_featurename_lvals = {}
    d_featurename_lvals_exponentiated_logreg = {}
    for featName in features_name:
        d_featurename_lvals[featName] = []
        d_featurename_lvals_exponentiated_logreg[featName] = []

    l_accuracy = []
    l_aucroc = []
    l_precision = []
    l_recall = []
    l_f1score = []

    l_numLabel1 = []
    l_propLabel1 = []

    for _ in xrange(n):
        l_ix_rand = np.random.choice(len(y), replace=True, size=len(y)).tolist()
        X_sample = X[l_ix_rand, :]
        y_sample = y[l_ix_rand]

        numLabel1 = np.sum(y_sample)
        l_numLabel1.append(numLabel1)
        propLabel1 = numLabel1 / len(y_sample) * 100
        l_propLabel1.append(propLabel1)

        if clf_name == 'Logistic regression':
            clf = LogisticRegression()
            clf.set_params(**optimised_clf_param)
            clf.fit(X_sample, y_sample)

            y_sample_pred = clf.predict(X_sample)
            scores_sample_pred = clf.predict_proba(X_sample)[:, 1]

            acc, aucroc, prec, recall, f1score = \
                get_acc_aucroc_prec_recall_f1score(y_sample, y_sample_pred, scores_sample_pred)

            l_accuracy.append(acc)
            l_aucroc.append(aucroc)
            l_precision.append(prec)
            l_recall.append(recall)
            l_f1score.append(f1score)

            featimportance = clf.coef_[0]
            for i in xrange(len(featimportance)):
                d_featurename_lvals[features_name[i]].append(featimportance[i])
                d_featurename_lvals_exponentiated_logreg[features_name[i]].append(math.exp(featimportance[i]))

        elif clf_name == 'Decision tree':
            clf = tree.DecisionTreeClassifier()
            clf.set_params(**optimised_clf_param)
            clf.fit(X_sample, y_sample)

            y_sample_pred = clf.predict(X_sample)
            scores_sample_pred = clf.predict_proba(X_sample)[:, 1]

            acc, aucroc, prec, recall, f1score = \
                get_acc_aucroc_prec_recall_f1score(y_sample, y_sample_pred, scores_sample_pred)

            l_accuracy.append(acc)
            l_aucroc.append(aucroc)
            l_precision.append(prec)
            l_recall.append(recall)
            l_f1score.append(f1score)

            featimportance = clf.feature_importances_
            for i in xrange(len(featimportance)):
                d_featurename_lvals[features_name[i]].append(featimportance[i])

        elif clf_name == 'Random forest':
            clf = RandomForestClassifier()
            clf.set_params(**optimised_clf_param)
            clf.fit(X_sample, y_sample)

            y_sample_pred = clf.predict(X_sample)
            scores_sample_pred = clf.predict_proba(X_sample)[:, 1]

            acc, aucroc, prec, recall, f1score = \
                get_acc_aucroc_prec_recall_f1score(y_sample, y_sample_pred, scores_sample_pred)

            l_accuracy.append(acc)
            l_aucroc.append(aucroc)
            l_precision.append(prec)
            l_recall.append(recall)
            l_f1score.append(f1score)

            featimportance = clf.feature_importances_
            for i in xrange(len(featimportance)):
                d_featurename_lvals[features_name[i]].append(featimportance[i])


    minNumbLabel1 = min(l_numLabel1)
    minPropLabel1 = min(l_propLabel1)
    maxNumbLabel1 = max(l_numLabel1)
    maxPropLabel1 = max(l_propLabel1)
    print ('\tLabel 1. Min numb: {0}, prop: {1}%. Max numb: {2}, prop: {3}%'.format(minNumbLabel1, minPropLabel1,
                                                                                  maxNumbLabel1, maxPropLabel1))

    # -- Confidence intervals
    alpha = 95
    print ('\tConfidence interval: {0}'.format(alpha))

    # -- Accuracy
    lower, upper = confint(np.asarray(l_accuracy))
    print ('\t\tAccuracy: {2} ({0}-{1})'.format(lower * 100, upper * 100, acc_calc * 100))

    # -- Precision
    lower, upper = confint(np.asarray(l_precision))
    print ('\t\tPrecision: {2} ({0}-{1})'.format(lower * 100, upper * 100, prec_calc * 100))

    # -- Recall
    lower, upper = confint(np.asarray(l_recall))
    print ('\t\tRecall: {2} ({0}-{1})'.format(lower * 100, upper * 100, recall_calc * 100))

    # -- F1-score
    lower, upper = confint(np.asarray(l_f1score))
    print ('\t\tF1-score: {2} ({0}-{1})'.format(lower * 100, upper * 100, fscore_calc * 100))

    # -- AUCROC
    lower, upper = confint(np.asarray(l_aucroc))
    print ('\t\tAUCROC: {2} ({0}-{1})'.format(lower, upper, aucroc_calc))

    print ('')

    if clf_name == 'Logistic regression':
        featimportance = optimised_clf.coef_[0]
        top = np.argpartition(featimportance, -4)[-4:]
        top_sorted = top[np.argsort(featimportance[top])]
        for i in top_sorted:
            print ('\t{0}: {1}. '.format(features_name[i], featimportance[i])),

            # -- CI
            lower, upper = confint(np.asarray(d_featurename_lvals[features_name[i]]))
            print ('{0}% CI: ({1}-{2})'.format(alpha, lower, upper))

            # -- Exponentiated
            lower, upper = confint(np.asarray(d_featurename_lvals_exponentiated_logreg[features_name[i]]))
            print('\t\tExponentiated {0}% CI: {1} ({2}-{3})'.format(alpha, math.exp(featimportance[i]), lower, upper))

    elif clf_name == 'Decision tree' or clf_name == 'Random forest':
        featimportance = optimised_clf.feature_importances_
        for i in xrange(len(featimportance)):
            print ('\t{0}: {1}'.format(features_name[i], featimportance[i])),

            # -- CI
            lower, upper = confint(np.asarray(d_featurename_lvals[features_name[i]]))
            print ('{0}% CI: ({1}-{2})'.format(alpha, lower, upper))

def ci_bootstrap_test(clf_name, optimised_clf, X, y, features_name,
                 acc_calc, prec_calc, recall_calc, fscore_calc, aucroc_calc, n=2000):
    np.random.seed(42)

    d_featurename_lvals = {}
    d_featurename_lvals_exponentiated_logreg = {}
    for featName in features_name:
        d_featurename_lvals[featName] = []
        d_featurename_lvals_exponentiated_logreg[featName] = []

    l_accuracy = []
    l_aucroc = []
    l_precision = []
    l_recall = []
    l_f1score = []

    for _ in xrange(n):
        l_ix_rand = np.random.choice(len(y), replace=True, size=len(y)).tolist()
        X_sample = X[l_ix_rand, :]
        y_sample = y[l_ix_rand]

        y_sample_pred = optimised_clf.predict(X_sample)
        scores_sample_pred = optimised_clf.predict_proba(X_sample)[:, 1]

        acc, aucroc, prec, recall, f1score = \
            get_acc_aucroc_prec_recall_f1score(y_sample, y_sample_pred, scores_sample_pred)

        l_accuracy.append(acc)
        l_aucroc.append(aucroc)
        l_precision.append(prec)
        l_recall.append(recall)
        l_f1score.append(f1score)

    # -- Confidence intervals
    alpha = 95
    print ('\tConfidence interval: {0}'.format(alpha))

    # -- Accuracy
    lower, upper = confint(np.asarray(l_accuracy))
    print ('\t\tAccuracy: {2} ({0}-{1})'.format(lower * 100, upper * 100, acc_calc * 100))

    # -- Precision
    lower, upper = confint(np.asarray(l_precision))
    print ('\t\tPrecision: {2} ({0}-{1})'.format(lower * 100, upper * 100, prec_calc * 100))

    # -- Recall
    lower, upper = confint(np.asarray(l_recall))
    print ('\t\tRecall: {2} ({0}-{1})'.format(lower * 100, upper * 100, recall_calc * 100))

    # -- F1-score
    lower, upper = confint(np.asarray(l_f1score))
    print ('\t\tF1-score: {2} ({0}-{1})'.format(lower * 100, upper * 100, fscore_calc * 100))

    # -- AUCROC
    lower, upper = confint(np.asarray(l_aucroc))
    print ('\t\tAUCROC: {2} ({0}-{1})'.format(lower, upper, aucroc_calc))


def run_gridsearchcv_Lasso(X, y):
    alphas = np.logspace(-4, -0.5, 30)
    alphas = np.append(alphas, 1.0)
    parameters = {'alpha': alphas,
                  'random_state':[42]
                  }

    clf = Lasso(max_iter=1000, fit_intercept=True)
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, refit=True, iid=True)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_estimator_.get_params()

def run_gridsearchcv_LogisticRegression(X, y):
    parameters = {'penalty': ['l1', 'l2'],
                  'class_weight': ['balanced'],
                  'solver': ['liblinear'],
                  'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                  'n_jobs': [-1],
                  'random_state':[42]
                  }

    clf = LogisticRegression(max_iter=1000, fit_intercept=True)
    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=5, refit=True, iid=True)
    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_estimator_.get_params()

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

from statsmodels.stats.proportion import proportion_confint
def calc_95CI_binomial_ruledev():
    # Number of correct prediction, number of instance were from each iteration

    numCorrectPrediction, numInstances = 48, 50
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 97, 100
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 50, 50
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 100, 100
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 48, 50
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 95, 100
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 98, 100
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 143, 150
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 146, 150
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    numCorrectPrediction, numInstances = 192, 200
    lower, upper = proportion_confint(numCorrectPrediction, numInstances, 0.05)   # 0.05: 95% CI
    print ('numCorrectPrediction: {0}, numInstances: {1}'.format(numCorrectPrediction, numInstances))
    print ('\t{0}, {1}'.format(lower, upper))

    done()

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

        # _, pval = ks_2samp(np.asarray(l_lvals[0]), np.asarray(l_lvals[1]))
        # print ('\tp-val: {0}\n'.format(pval))

        _, pval = mannwhitneyu(np.asarray(l_lvals[0]), np.asarray(l_lvals[1]))
        print ('\tp-val (Mann-Whitney U test): {0}\n'.format(pval))

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
    f = open(inFile, 'r')

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

def print_env():
    print ('scikit-learn {0}'.format(sklearn.__version__))
    print ('SciPy {0}'.format(scipy.__version__))
    print ('NumPy {0}'.format(np.__version__))
    done()


if __name__ == '__main__':
    # print_env()

    resultFolder = 'Results/'
    cpickleFolder = 'cpickle/'

    LOAD_PREVIOUS = False
    RETRAINED = False

    print ('[0] Load previous trained classifiers')
    print ('[1] Retrained the classifiers on your dataset')
    ipt = input("Option: ")

    if ipt == 0: LOAD_PREVIOUS = True
    elif ipt == '1': RETRAINED = True
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

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        numbLabel1 = np.sum(y_train)
        lenytrain = len(y_train)
        propLabel1 = numbLabel1 / lenytrain * 100
        print ('\ny_train. len: {0}, numb label 1: {1} ({2}%)\n'.format(lenytrain, numbLabel1, propLabel1))

        # -- Classifiers
        # ----------------------
        print ('\nLogistic Regression')
        print ('------------')
        # optimised_LR, optimised_params_LR = run_gridsearchcv_LogisticRegression(X_train, y_train)
        # print ('{0}'.format(optimised_params_LR))
        # dump_var(optimised_LR, resultFolder + cpickleFolder + 'optimised_LR')
        # dump_var(optimised_params_LR, resultFolder + cpickleFolder + 'optimised_params_LR')

        optimised_LR = load_var(resultFolder + cpickleFolder + 'optimised_LR')
        optimised_params_LR = load_var(resultFolder + cpickleFolder + 'optimised_params_LR')

        print ('[ Training set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_LR, y_train, X_train)
        ci_bootstrap_train('Logistic regression', optimised_LR, optimised_params_LR, X_train, y_train, features_name,
                     acc, prec, recall, fscore, aucroc)

        print ('[ Test set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_LR, y_test, X_test)
        ci_bootstrap_test('Logistic regression', optimised_LR, X_test, y_test, features_name,
                           acc, prec, recall, fscore, aucroc)


        print ('\nDecision Tree')
        print ('------------')
        # optimised_DT, optimised_params_DT = run_gridsearchcv_DTClassifier(X_train, y_train)
        # print ('{0}'.format(optimised_params_DT))
        # dump_var(optimised_DT, resultFolder + cpickleFolder + 'optimised_DT')
        # dump_var(optimised_params_DT, resultFolder + cpickleFolder + 'optimised_params_DT')

        optimised_DT = load_var(resultFolder + cpickleFolder + 'optimised_DT')
        optimised_params_DT = load_var(resultFolder + cpickleFolder + 'optimised_params_DT')

        print ('[ Training set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_DT, y_train, X_train)
        ci_bootstrap_train('Decision tree', optimised_DT, optimised_params_DT, X_train, y_train, features_name,
                     acc, prec, recall, fscore, aucroc)

        print ('[ Test set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_DT, y_test, X_test)
        ci_bootstrap_test('Decision tree', optimised_DT, X_test, y_test, features_name,
                           acc, prec, recall, fscore, aucroc)


        print ('\nRandom Forest')
        print ('------------')
        # optimised_RF, optimised_params_RF = run_gridsearchcv_RFClassifier(X_train, y_train)
        # print ('{0}'.format(optimised_params_RF))
        # dump_var(optimised_RF, resultFolder + cpickleFolder + 'optimised_RF')
        # dump_var(optimised_params_RF, resultFolder + cpickleFolder + 'optimised_params_RF')

        optimised_RF = load_var(resultFolder + cpickleFolder + 'optimised_RF')
        optimised_params_RF = load_var(resultFolder + cpickleFolder + 'optimised_params_RF')

        print ('[ Training set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_RF, y_train, X_train)
        ci_bootstrap_train('Random forest', optimised_RF, optimised_params_RF, X_train, y_train, features_name,
                     acc, prec, recall, fscore, aucroc)

        print ('[ Test set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_RF, y_test, X_test)
        ci_bootstrap_test('Random forest', optimised_RF, X_test, y_test, features_name,
                           acc, prec, recall, fscore, aucroc)


        # -- Calibration plot
        l_clf_modelname = [(optimised_LR, 'Logistic regression'),
                           (optimised_DT, 'Decision tree'),
                           (optimised_RF, 'Random forest')]

        calibration(X_test, y_test, l_clf_modelname)



        # calc_95CI_binomial_ruledev()    # to calculate the 95% CI based on the Binomial proportion
        # -- feature importance based on LASSO
        print ('\nLasso')
        print ('------------')
        # optimised_Lasso, optimised_params_Lasso = run_gridsearchcv_Lasso(X_train, y_train)
        # print ('{0}'.format(optimised_params_Lasso))
        # dump_var(optimised_Lasso, resultFolder + cpickleFolder + 'optimised_Lasso')
        # dump_var(optimised_params_Lasso, resultFolder + cpickleFolder + 'optimised_params_Lasso')

        optimised_Lasso = load_var(resultFolder + cpickleFolder + 'optimised_Lasso')
        optimised_params_Lasso = load_var(resultFolder + cpickleFolder + 'optimised_params_Lasso')

        featimportance = optimised_Lasso.coef_

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

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        numbLabel1 = np.sum(y_train)
        lenytrain = len(y_train)
        propLabel1 = numbLabel1 / lenytrain * 100
        print ('\ny_train. len: {0}, numb label 1: {1} ({2}%)\n'.format(numbLabel1, lenytrain, propLabel1))


        # -- Classifiers
        # ----------------------
        print ('\nLogistic Regression')
        print ('------------')
        optimised_LR, optimised_params_LR = run_gridsearchcv_LogisticRegression(X_train, y_train)
        print ('{0}'.format(optimised_params_LR))
        dump_var(optimised_LR, resultFolder + cpickleFolder + 'optimised_LR_own')
        dump_var(optimised_params_LR, resultFolder + cpickleFolder + 'optimised_params_LR_own')

        optimised_LR = load_var(resultFolder + cpickleFolder + 'optimised_LR_own')
        optimised_params_LR = load_var(resultFolder + cpickleFolder + 'optimised_params_LR_own')

        print ('[ Training set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_LR, y_train, X_train)
        ci_bootstrap_train('Logistic regression', optimised_LR, optimised_params_LR, X_train, y_train, features_name,
                     acc, prec, recall, fscore, aucroc)

        print ('[ Test set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_LR, y_test, X_test)
        ci_bootstrap_test('Logistic regression', optimised_LR, X_test, y_test, features_name,
                           acc, prec, recall, fscore, aucroc)


        print ('\nDecision Tree')
        print ('------------')
        optimised_DT, optimised_params_DT = run_gridsearchcv_DTClassifier(X_train, y_train)
        print ('{0}'.format(optimised_params_DT))
        dump_var(optimised_DT, resultFolder + cpickleFolder + 'optimised_DT_own')
        dump_var(optimised_params_DT, resultFolder + cpickleFolder + 'optimised_params_DT_own')

        optimised_DT = load_var(resultFolder + cpickleFolder + 'optimised_DT_own')
        optimised_params_DT = load_var(resultFolder + cpickleFolder + 'optimised_params_DT_own')

        print ('[ Training set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_DT, y_train, X_train)
        ci_bootstrap_train('Decision tree', optimised_DT, optimised_params_DT, X_train, y_train, features_name,
                     acc, prec, recall, fscore, aucroc)

        print ('[ Test set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_DT, y_test, X_test)
        ci_bootstrap_test('Decision tree', optimised_DT, X_test, y_test, features_name,
                           acc, prec, recall, fscore, aucroc)


        print ('\nRandom Forest')
        print ('------------')
        optimised_RF, optimised_params_RF = run_gridsearchcv_RFClassifier(X_train, y_train)
        print ('{0}'.format(optimised_params_RF))
        dump_var(optimised_RF, resultFolder + cpickleFolder + 'optimised_RF_own')
        dump_var(optimised_params_RF, resultFolder + cpickleFolder + 'optimised_params_RF_own')

        optimised_RF = load_var(resultFolder + cpickleFolder + 'optimised_RF')
        optimised_params_RF = load_var(resultFolder + cpickleFolder + 'optimised_params_RF_own')

        print ('[ Training set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_RF, y_train, X_train)
        ci_bootstrap_train('Random forest', optimised_RF, optimised_params_RF, X_train, y_train, features_name,
                     acc, prec, recall, fscore, aucroc)

        print ('[ Test set ]')
        acc, prec, recall, fscore, aucroc = evaluate_test(optimised_RF, y_test, X_test)
        ci_bootstrap_test('Random forest', optimised_RF, X_test, y_test, features_name,
                           acc, prec, recall, fscore, aucroc)

        done()

