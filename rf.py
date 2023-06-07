"""
This file contains a hyperparameter tuning of a random forest
"""
import dataUtilities as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score
import numpy as np
import sys
import joblib

def testModel(clfs, xTest, yTest):
    """
    Computes and prints the test score of a given classifier
    :param clfs: trained sklearn classifier object
    :param xTest: ndarray of test data
    :param yTest: ndarray of test labels
    :return: None
    """
    bestIndex = np.argmax(clfs['estimators']['test_score'])
    clf = clfs['estimators'][bestIndex]
    pred = clf.predict(xTest)
    print('Test F1-Score: ' + f1_score(yTest, pred))
    return

def classify(xTrain, yTrain, param):
    """
    Gridsearch for the best random forest parameters using cross validation
    :param xTrain: ndarray containing the training data
    :param yTrain: ndarray containing the training labels
    :param param: ndarray of interval definition for parameter gridsearch
    :return: sklearn clf object of best random forest
    """
    best = {'score': 0, 'estimators': []}
    params = [0, 0, 0]
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for j in range((param*50)+1, (1+param) * 50):
        for k in range(1, 25):
            for n in range(50, 250):
                clf = RandomForestClassifier(n_estimators=j, bootstrap=True,
                                             max_depth=k, criterion='entropy',
                                             max_samples=n, n_jobs=-1)
                scores = cross_validate(clf, xTrain, yTrain, scoring='f1_macro', return_estimator=True,
                                        cv=skf)
                if np.amax(scores['test_score']) > best['score']:
                    best['score'] = np.amax(scores['test_score'])
                    best['estimators'] = scores
                    params[0] = j
                    params[1] = k
                    params[2] = n
        print("F1-Score: %0.2f (+/- %0.2f)" % (best['estimators']['test_score'].mean(), best['estimators']['test_score'].std() * 2))
    print("Final F1-Score: %0.2f (+/- %0.2f)" % (best['estimators']['test_score'].mean(), best['estimators']['test_score'].std() * 2))
    return best

def example(param):
    """
    Runs an example for Random Forest
    :return: None
    """
    inputParam = param
    dataset = dt.loadReplacedSet()
    training, _, test = dt.splitFrame(dataset)
    train = training.drop(columns=['Behandlungskategorie']).values
    testdata = test.drop(columns=['Behandlungskategorie']).values
    bestClassifier = classify(train, training['Behandlungskategorie'].values, int(inputParam))
    print('Training finished')
    testModel(bestClassifier, testdata, test['Behandlungskategorie'].values)
    return
