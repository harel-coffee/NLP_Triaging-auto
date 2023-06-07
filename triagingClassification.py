"""
In this file a kNN ensemble approach is used. One kNN classifier is trained for each set of flags
(black, blue, yellow, red, undetermined, custom scores)
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import dataUtilities as dt

def testModel(clfs, xTest, yTest):
    """
    Computes and prints the test score for the best classifier
    :param clfs: dict of trained classifiers using sklearns cross validation
    :param xTest: ndarray containing the test data
    :param yTest: ndarray containing the test labels
    :return: None
    """
    bestIndex = np.argmax(clfs['estimators']['test_score'])
    clf = clfs['estimators'][bestIndex]
    pred = clf.predict(xTest)
    print('Test Accuracy: ' + accuracy_score(yTest, pred))
    print('Test F1-Score: ' + f1_score(yTest, pred))
    return

def createKNNClassifiers(xTrain, yTrain, listOfFlags, neighbours=np.ones(shape=6)+5):
    """
    Creates and trains the kNN classifiers for each set of flags
    :param xTrain: ndarray containing the training data
    :param yTrain: ndarray containing the training labels
    :param listOfFlags: list of list of strings containing the flags/items
    :param neighbours: ndarray containing the number of considered neighbours
    :return: dict of trained kNN classifiers
    """
    dictOfClfs = {'red': None, 'yellow': None,
                  'blue': None, 'black': None,
                  'undetermined': None, 'custom': None}
    for index, flags in listOfFlags:
        clf = KNeighborsClassifier(n_neighbors=neighbours[index], n_jobs=-1, weights='distance')
        clf.fit(xTrain[flags], yTrain)
        dictOfClfs[dictOfClfs.keys()[index]] = clf
    return dictOfClfs

def metric(clfResults, weights=np.ones(shape=9, dtype=float)):
    """
    Computes the final classification result by majority vote
    :param clfResults: ndarray containing the predictions of each classifier
    :param weights: ndarray containing the weights for each class
    :return: ndarray, ndarray containing the number of votes, containing the voted for label
    """
    votes = np.zeros(shape=3, dtype=float)
    t1 = np.argwhere(clfResults == 1)
    t2 = np.argwhere(clfResults == 2)
    t6 = np.argwhere(clfResults == 6)
    vT1 = np.sum(np.multiply(clfResults[t1], weights[t1]))
    vT2 = np.sum(np.multiply(clfResults[t2], weights[t2]))
    vT6 = np.sum(np.multiply(clfResults[t6], weights[t6]))
    votes[0] = vT1
    votes[1] = vT2
    votes[2] = vT6
    decision = np.argmax(votes)
    if decision == 0:
        label = 1
    elif decision == 1:
        label = 2
    elif decision == 2:
        label = 6
    else:
        posLabel = np.array([1, 2, 6])
        label = posLabel[np.random.randint(0, 3, 1)]
        print('Label guessed')
    return votes, label

def onehotEncoding(true_label, pred_label):
    """
    Returns one hot encoded labels for further processing
    :param true_label: ndarray of true labels
    :param pred_label: ndarray of predicted labels
    :return: ndarray of one hot encoded labels
    """
    present_labels = np.unique(true_label)
    true_hot = np.zeros(shape=(true_label.shape[0], present_labels.shape[0]))
    pred_hot = np.zeros(shape=(pred_label.shape[0], present_labels.shape[0]))
    for i in range(true_hot.shape[0]):
        true_hot[np.argwhere(present_labels == true_hot[i])] = 1
        pred_hot[np.argwhere(present_labels == pred_hot[i])] = 1
    return true_hot, pred_hot

red = ['LWKNIH0410', 'LWKWKC03_0110', 'LWKWKC03_0210', 'LWKWKC03_0310',
           'LWKWKC03_0410', 'LWKWKC03_0510', 'LWKWKC03_0610', 'LWKWKC03_0710',
           'LWKWKC03_0810', 'LWKWKC03_0910', 'LWKWKC03_1010', 'LWKWKC03_1110',
           'LWKWKC03_1510', 'LWKWKC03_1610', 'LWKWKC03_1710', 'LWKWKC03_1810',
           'LWKWKC03_1910', 'LWKWKC03_2010', 'LWKWKC03_2110', 'LWKNIH09_110',
           'LWKNIH09_210', 'LWKNIH09_1_110', 'LWKNIH1110']
yellow = ['LWKNIH2210', 'LWKNIH2310', 'LWKNIH2410', 'LWKNIH2510',
              'LWKNIH2610', 'LWKNIH2710', 'LWKNIH2810', 'LWKNIH2910',
              'LWKNIH3010', 'LWKNIH3110', 'LWKNIH3210', 'LWKNIH3310',
              'LWKNIH3810']
blue = ['GSCWORK1410', 'GSCCOPSOQ0110', 'GSCCOPSOQ0210', 'GSCCOPSOQ0310',
            'GSCCOPSOQ0410', 'GSCCOPSOQ0510', 'GSCCOPSOQ0610', 'GSCCOPSOQ0710',
            'GSCCOPSOQ0810', 'GSCCOPSOQ0910', 'GSCCOPSOQ1010', 'GSCCOPSOQ1110',
            'GSCCOPSOQ1210', 'GSCCOPSOQ1310', 'GSCCOPSOQ1410', 'GSCCOPSOQ1510',
            'GSCCOPSOQ1610', 'GSCCOPSOQ1710', 'GSCCOPSOQ1810', 'GSCCOPSOQ1910',
            'GSCCOPSOQ2010', 'GSCCOPSOQ2110', 'GSCCOPSOQ2210', 'GSCCOPSOQ2310',
            'GSCCOPSOQ2410', 'GSCCOPSOQ2510']
black = ['GSCPCQ0110', 'GSCPCQ01_110', 'GSCPCQ0210', 'GSCPCQ0310', 'GSCPCQ03_110',
             'GSCPCQ03_210', 'GSCPCQ0410', 'GSCPCQ04_110', 'GSCPCQ0610',
             'GSCPCQ0710', 'GSCPCQ0810', 'GSCPCQ0910', 'GSCPCQ1010'] #'GSCPCQ0510', rausgenommen weil timestamp

undetermined = ['LWKWKC03_1410', 'LWKWKC03_2210', 'LWKWKC03_2310',
                    'LWKWKC03_1310', 'LWKNIH09_310', 'LWKWKC03_1210',
                    'LWKNIH09_410', 'LWKWKC03_2410']
customScores = ['Stress_arbeit_privatleben', 'alcohol_drugs', 'Depression_Index', 'Schlafqualität']
flags = [red, yellow, blue, black, undetermined, customScores]

# knn ensemble from here löschen
train, val, test = dt.loadAsFrame([item for sublist in flags for item in sublist])
flags.append(customScores)
yTrain = train['Behandlungskategorie']
xTrain = train.drop(columns=['Behandlungskategorie'])
yVal = val['Behandlungskategorie']
xVal = val.drop(columns=['Behandlungskategorie'])
yTest = test['Behandlungskategorie']
xTest = test.drop(columns=['Behandlungskategorie'])
bestClfs = []
bestScores = []
for flagSet in flags:
    tempBestScore = 0
    tempBestClf = None
    for i in range(1, 100):
        flagClf = KNeighborsClassifier(n_neighbors=i, weights='distance', n_jobs=8)
        flagClf.fit(xTrain[flagSet], yTrain)
        y_true = yVal
        y_pred = flagClf.predict(xVal[flagSet])
        fscore = f1_score(y_true, y_pred, average='weighted')
        if fscore > tempBestScore:
            tempBestClf = flagClf
            tempBestScore = fscore
    bestClfs.append(tempBestClf)
    bestScores.append(tempBestScore)
predictions = np.zeros(shape=xVal.shape)
for index, flag in enumerate(flags):
    predictions[:, index] = bestClfs[index].predict(xVal[flag])
pred_labels = []
for i in range(0, predictions.shape[0]):
    pred, counts = np.unique(predictions[i, :], return_counts=True)
    majority = pred[np.argmax(counts)]
    pred_labels.append(majority)
print(f1_score(y_true=yVal, y_pred=np.asarray(pred_labels), average='weighted'))
exit('Done! Have a nice day.')
