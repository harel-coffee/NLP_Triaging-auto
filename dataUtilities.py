import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def loadTreatmentData(pathToTreatmentData):
    """
    Loads the treatment catagories into a DataFrame
    :param pathToTreatmentData: string path to treatment data
    :return: DataFrame containing the treatment information
    """
    data = pd.read_csv(pathToTreatmentData, sep=';', usecols=[0, 13], index_col=0)
    return data

def loadUMCGData(pathToUMCGData):
    """
    Loads the raw dataset into a DataFrame
    :param pathToUMCGData: string path to data
    :return: DataFrame containing the dataset
    """
    data = pd.read_csv(pathToUMCGData, sep=';', low_memory=False, decimal=',')
    return data

def depressionScore(dataset):
    """
           Helper function to compute the custom score "Depression score"
           :param dataset: DataFrame containing the patients and at least four items
           :return: DataFrame containing the "Depression score"
           """
    meanValues = dataset.loc[:, ['LWKNIH2210', 'LWKNIH2310', 'LWKNIH2410', 'LWKNIH2510']].mean(axis=1)
    dataset['Depression_Index'] = meanValues
    return dataset


def sleepQualityScore(dataset):
    """
        Helper function to compute the custom score "Schlafqualität"
        :param dataset: DataFrame containing the patients and at least three items
        :return: DataFrame containing the "Schlafqualität score"
    """
    meanValues = dataset.loc[:, ['LWKNIH2710', 'LWKNIH2810', 'LWKNIH2910']].mean(axis=1)
    dataset['Schlafqualität'] = meanValues
    return dataset

def drugScore(dataset):
    """
        Helper function to compute the custom score "Drogenmissbrauch"
        :param dataset: DataFrame containing the patients and at least two items
        :return: DataFrame containing the "Drogenmissbrauchs score"
    """
    meanValues = dataset.loc[:, ['LWKNIH3210', 'LWKNIH3310']].mean(axis=1)
    dataset['alcohol_drugs'] = meanValues
    return dataset

def stressScore(dataset):
    """
    Helper function to compute the custom score "Stress"
    :param dataset: DataFrame containing the patients and at least two items
    :return: DataFrame containing the "Stress score"
    """
    meanValues = dataset.loc[:, ['GSCCOPSOQ2210', 'GSCCOPSOQ2310']].mean(axis=1)
    dataset['Stress_arbeit_privatleben'] = meanValues
    return dataset

#all black and blue flags, because no work means items not applicable
def fillWork(dataset):
    """
    Helper function to fill in the omitted blue and black flags with -1
    :param dataset: DataFrame containing all patients
    :return: DataFrame containing filled omitted values
    """
    blue = ['GSCWORK1410', 'GSCCOPSOQ0110', 'GSCCOPSOQ0210', 'GSCCOPSOQ0310',
            'GSCCOPSOQ0410', 'GSCCOPSOQ0510', 'GSCCOPSOQ0610', 'GSCCOPSOQ0710',
            'GSCCOPSOQ0810', 'GSCCOPSOQ0910', 'GSCCOPSOQ1010', 'GSCCOPSOQ1110',
            'GSCCOPSOQ1210', 'GSCCOPSOQ1310', 'GSCCOPSOQ1410', 'GSCCOPSOQ1510',
            'GSCCOPSOQ1610', 'GSCCOPSOQ1710', 'GSCCOPSOQ1810', 'GSCCOPSOQ1910',
            'GSCCOPSOQ2010', 'GSCCOPSOQ2110', 'GSCCOPSOQ2210', 'GSCCOPSOQ2310',
            'GSCCOPSOQ2410', 'GSCCOPSOQ2510']
    black = ['GSCPCQ0110', 'GSCPCQ01_110', 'GSCPCQ0210', 'GSCPCQ0310', 'GSCPCQ03_110',
             'GSCPCQ03_210', 'GSCPCQ0410', 'GSCPCQ04_110', 'GSCPCQ0610',
             'GSCPCQ0710', 'GSCPCQ0810', 'GSCPCQ0910', 'GSCPCQ1010']
    cols = blue + black
    dataset[cols] = dataset[cols].fillna(value=-1)
    return dataset

def fillRisk(dataset):
    """
    Helper function to fill in the omitted values for risk flags with -1
    :param dataset: DataFrame containing the patients and all items
    :return: DataFrame with filled omitted values
    """
    omittedCols = ['LWKWKC03_1110', 'LWKWKC03_1210', 'LWKWKC03_1310', 'LWKWKC03_1410',
                   'LWKWKC03_1510', 'LWKWKC03_1610', 'LWKWKC03_1710', 'LWKWKC03_1810',
                   'LWKWKC03_1910', 'LWKWKC03_2010', 'LWKWKC03_2110', 'LWKWKC03_2210',
                   'LWKWKC03_2310', 'LWKWKC03_2410', 'LWKNIH09_110', 'LWKNIH09_210',
                   'LWKNIH09_310', 'LWKNIH09_410']
    dataset.loc[dataset['LWKWKC03_1010'] == 0, omittedCols] = -1
    dataset['LWKNIH09_1_110'] = dataset['LWKNIH09_1_110'].fillna(-1)
    return dataset


def fillOmittedValues(dataset):
    """
    Helper function to fill the omitted values in the questionnaires
    :param dataset: DataFrame containing the patients and all items
    :return: DataFrame with filled omitted values
    """
    datasetRisk = fillRisk(dataset)
    datasetWork = fillWork(datasetRisk)
    return datasetWork

def dimensionalityReduction(umcgData):
    """
    Helper function to compute the custom scores. The underlying items are removed
    :param umcgData: DataFrame containing the patients and all items
    :return: DataFrame containing the custom scores and not containing the underlying items
    """
    data = depressionScore(umcgData)
    data = sleepQualityScore(data)
    data = drugScore(data)
    data = stressScore(data)
    return data

def comparePAID(umcgData, treatmentData):
    """
    Helper function for verifying data integrity
    :param umcgData: DataFrame containing patient data
    :param treatmentData: DataFrame containing the treatment categories
    :return: bool true if passed, false if failed
    """
    check = False
    treatPAID, treatCounts = np.unique(treatmentData.index, return_counts=True)
    umcgPAID, umcgCounts = np.unique(umcgData.index, return_counts=True)
    interPAID = np.intersect1d(treatPAID, umcgPAID)
    print('Intersection length: ' + str(len(interPAID)))
    print('Treatment length: ' + str(len(treatPAID)))
    print('UMCG length: ' + str(len(umcgPAID)))
    return check

def checkTreatment(treatmentData):
    """
    Computes and prints the treatment category counts
    :param treatmentData: DataFrame containing the patients with treatment category
    :return: None
    """
    treatments, counts = np.unique(treatmentData['Treatment category'], return_counts=True)
    print('Treatment categories: ' + str(treatments))
    print('Assignment counts: ' + str(counts))
    return

def addTreatmentCategory(umcgData, treatmentData):
    """
    Not needed anymore
    Adds the treatment category for each patient in a new column
    :param umcgData: DataFrame containing the patient data
    :param treatmentData: DataFrame containing the treatment path code
    :return: DataFrame including the treatment categories
    """
    umcgData['Treatment category'] = 0
    for index, row in treatmentData.iterrows():
        umcgData.loc[index, 'Treatment category'] = row['Treatment category']
    return umcgData

def showOutcomeScales(umcg):
    """
    Prints the mean and the variance of the outcome measures Pain Disability Index, RTF Impact Score, and EQ5D at T0
    :param umcg: DataFrame containing the dataset
    :return: None
    """
    outcomes = umcg[['RTFIMPACT_T0', 'EQ5DNLT0', 'PDITotalT0']]
    print(outcomes.dtypes)
    print(outcomes.mean())
    print(outcomes.var())
    return

def convertColToFloat(col):
    """
    Helper function to convert columns from the raw dataset
    :param col: DataFrame containing one column
    :return: DataFrame containing the column as type float
    """
    floatCol = np.zeros(shape=(col.shape[0]), dtype=float)
    for i in range(0, col.shape[0]):
        try:
            tempFloat = float(col[i])
            floatCol[i] = tempFloat
        except Exception as exc:
            print(exc)
            split = col[i].split(',')
            floatCol[i] = float(split[0]) + np.divide(float(split[1]), 10)
    return floatCol

def analyseTriaging(clf, data):
    """
    Computes the majority of labels in each cluster
    :param clf: sklearn clustering object
    :param data: ndarray containing the data used for clustering
    :return: ndarray with labels for clusters
    """
    triaging = np.zeros(shape=(clf.n_components, 6))
    for i in range(data.shape[0]):
        currSample = data[i, :-1]
        cluster = clf.predict(currSample.reshape(1, -1))
        triaging[cluster[0], int(data[i, -1])-1] = triaging[cluster[0], int(data[i, -1])-1] + 1
    return triaging


def getDataOfInterest(umcgPath, header):
    """
    Loads the items of interest for patients from the dataset who has treatment categories
    :param umcgPath: string path to dataset
    :param header: list of strings containing the items of interest
    :return: DataFrame containing the data of interest
    """
    umcg = loadUMCGData(umcgPath)
    dataOfInterest = umcg[header]
    dataOfInterest = dataOfInterest.replace(r'^\s*$', np.nan, regex=True)
    dataOfInterest = dataOfInterest[np.logical_not(dataOfInterest['Behandlungskategorie'].isna())]
    dataOfInterest = dataOfInterest.dropna(axis=0)
    dataOfInterest = dataOfInterest.values
    dataOfInterest = dataOfInterest.astype(float)
    return dataOfInterest

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.means_]#[kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = len(kmeans.n_components)#kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape
    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]],
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    return np.multiply(BIC, -1)

def cleanData(umcgPath):
    """
    Loads the data and cleans it
    :param umcgPath: string containing the path to the dataset
    :return: DataFrame containing the data
    """
    umcg = loadUMCGData(umcgPath)
    umcg = umcg.replace(r'^\s*$', np.nan, regex=True)
    umcg = umcg.replace(r'\?', np.nan, regex=True)
    umcg[['PDITotalT0', 'PDITotalT1',
         'PDITotalT2', 'PDITotalT3',
         'PDITotalT4', 'EQ5DNLT1',
         'EQ5DNLT2', 'EQ5DNLT3', 'EQ5DNLT4',
         'Age']] = umcg[['PDITotalT0', 'PDITotalT1',
         'PDITotalT2', 'PDITotalT3',
         'PDITotalT4', 'EQ5DNLT1',
         'EQ5DNLT2', 'EQ5DNLT3', 'EQ5DNLT4',
         'Age']].replace('\,', '.', regex=True)
    umcg['OVLDAT'] = pd.to_datetime(umcg['OVLDAT'])
    umcg['DQRECEIVEDDAT12'] = pd.to_datetime(umcg['DQRECEIVEDDAT12'])
    umcg['DQRECEIVEDDAT13'] = pd.to_datetime(umcg['DQRECEIVEDDAT13'])
    umcg['EVAL01DAT11'] = pd.to_datetime(umcg['EVAL01DAT11'])
    umcg['GSCPCQ0512'] = pd.to_datetime(umcg['GSCPCQ0512'])
    umcg['GSCPCQ0510'] = pd.to_datetime(umcg['GSCPCQ0510'])
    umcg['GSCPCQ0514'] = pd.to_datetime(umcg['GSCPCQ0514'])
    umcg['LWKWKC02_NL110'] = umcg['LWKWKC02_NL110'].values.astype(str)
    umcg['LWKWCK02_RMA110'] = umcg['LWKWCK02_RMA110'].values.astype(str)
    umcg['LWKWCK02_RMA210'] = umcg['LWKWCK02_RMA210'].values.astype(str)
    umcg['LWKWKC02_INT110'] = umcg['LWKWKC02_INT110'].values.astype(str)
    umcg['LWKWKC02_INT210'] = umcg['LWKWKC02_INT210'].values.astype(str)
    umcg['LWKWKC02_PL110'] = umcg['LWKWKC02_PL110'].values.astype(str)
    umcg['LWKWKC02_PL210'] = umcg['LWKWKC02_PL210'].values.astype(str)
    umcg['LWKWKC02_PSY110'] = umcg['LWKWKC02_PSY110'].values.astype(str)
    umcg['LWKWKC02_PSY210'] = umcg['LWKWKC02_PSY210'].values.astype(str)
    umcg['LWKWKC04_6_110'] = umcg['LWKWKC04_6_110'].values.astype(str)
    umcg['LWKNIH37_110'] = umcg['LWKNIH37_110'].values.astype(str)
    umcg['LWKWKC04_6_110'] = umcg['LWKWKC04_6_110'].values.astype(str)
    umcg['LWKNIH39_11_110'] = umcg['LWKNIH39_11_110'].values.astype(str)
    umcg['LWKNIH39_11_111'] = umcg['LWKNIH39_11_111'].values.astype(str)
    umcg['LWKWKC04_6_112'] = umcg['LWKWKC04_6_112'].values.astype(str)
    umcg['LWKNIH39_11_112'] = umcg['LWKNIH39_11_112'].values.astype(str)
    umcg['LWKNIH39_11_113'] = umcg['LWKNIH39_11_113'].values.astype(str)
    umcg['LWKWKC04_6_114'] = umcg['LWKWKC04_6_114'].values.astype(str)
    umcg['LWKNIH39_11_114'] = umcg['LWKNIH39_11_114'].values.astype(str)
    umcg['EVAL05_9_011'] = umcg['EVAL05_9_011'].values.astype(str)
    umcg['GSCPCQ03_2_114'] = umcg['GSCPCQ03_2_114'].values.astype(str)
    umcg['Age'] = umcg['Age'].values.astype(float)
    cols = list(umcg)
    convertedCols = ['OVLDAT', 'LWKWKC02_NL110', 'LWKWCK02_RMA110', 'LWKWCK02_RMA210',
                       'LWKWKC02_INT110', 'LWKWKC02_INT210', 'LWKWKC02_PL110',
                       'LWKWKC02_PL210', 'LWKWKC02_PSY110', 'LWKWKC02_PSY210',
                       'LWKNIH37_110', 'LWKWKC04_6_110', 'LWKNIH39_11_110',
                       'LWKNIH39_11_111', 'DQRECEIVEDDAT12', 'LWKWKC04_6_112',
                       'LWKNIH39_11_112', 'DQRECEIVEDDAT13', 'LWKNIH39_11_113',
                       'LWKWKC04_6_114', 'LWKNIH39_11_114', 'EVAL01DAT11',
                       'EVAL05_9_011', 'EVAL02DAT12', 'GSCPCQ0510', 'GSCPCQ0512',
                       'GSCPCQ03_2_114', 'GSCPCQ0514', 'Age', 'PDITotalT0',
                       'PDITotalT1', 'EQ5DNLT1', 'PDITotalT2', 'EQ5DNLT2', 'PDITotalT3',
                       'EQ5DNLT3', 'PDITotalT4', 'EQ5DNLT4']
    for entry in convertedCols:
        try:
            cols.remove(entry)
        except Exception as exc:
            print(entry)
    umcg[cols] = umcg[cols].values.astype(float)
    return umcg

def splitFrame(dataset):
    """
    Splits the dataset into training, validation, and test data in a stratified fassion
    :param dataset: DataFrame containing the patients and items
    :return: Tuple DataFrames for training, validating and testing
    """
    data = dataset.drop(columns=['Behandlungskategorie'])
    datacols = list(data.columns)
    label = dataset['Behandlungskategorie']
    xTrain, xOther, yTrain, yOther = train_test_split(data.values, label.values,
                                                    random_state=42, test_size=0.25,
                                                    stratify=label)
    xVal, xTest, yVal, yTest = train_test_split(xOther, yOther,
                                                random_state=42, test_size=0.4,
                                                stratify=yOther)
    train = pd.DataFrame(data=xTrain, columns=datacols, dtype=float)
    train['Behandlungskategorie'] = yTrain
    val = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
    val['Behandlungskategorie'] = yVal
    test = pd.DataFrame(data=xTest, columns=datacols, dtype=float)
    test['Behandlungskategorie'] = yTest
    return train, val, test

def splitFrameCrossval(dataset):
    """
    Splits the dataset into four cross validation sets in a stratified fassion
    :param dataset: DataFrame containing the patients and flags
    :return: list of lists containing training and validation data pairs, test data
    """
    data = dataset.drop(columns=['Behandlungskategorie'])
    datacols = list(data.columns)
    label = dataset['Behandlungskategorie']
    xRest, xVal, yRest, yVal = train_test_split(data.values, label.values,
                                                    random_state=42, test_size=0.33,
                                                    stratify=label)
    scaler = StandardScaler()
    scaler.fit(xRest)
    xRest = scaler.transform(xRest)
    xVal = scaler.transform(xVal)
    val = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
    val['Behandlungskategorie'] = yVal
    crossValSets = []
    for i in range(0, 4):
        xTrain, xVal, yTrain, yVal = train_test_split(xRest, yRest, test_size=0.25, stratify=yRest)
        xTrain = pd.DataFrame(data=xTrain, columns=datacols, dtype=float)
        xTrain['Behandlungskategorie'] = yTrain
        xVal = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
        xVal['Behandlungskategorie'] = yVal
        crossValSets.append([xTrain, xVal])
    return crossValSets, val

def splitFrameCrossvalCustom(dataset):
    """
    Splits the dataset into four cross validation sets in a stratified fassion
    :param dataset: DataFrame containing the patients and flags
    :return: list of lists containing training and validation data pairs, test data
    """
    data = dataset.drop(columns=['Behandlungskategorie'])
    datacols = list(data.columns)
    label = dataset['Behandlungskategorie']
    xRest, xVal, yRest, yVal = train_test_split(data.values, label.values,
                                                    random_state=42, test_size=0.33,
                                                    stratify=label)
    scaler = StandardScaler()
    scaler.fit(xRest)
    xRest = scaler.transform(xRest)
    xVal = scaler.transform(xVal)
    test = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
    test['Behandlungskategorie'] = yVal
    crossValSets = []
    for i in range(0, 4):
        xTrain, xVal, yTrain, yVal = train_test_split(xRest, yRest, test_size=0.25, stratify=yRest, random_state=42)
        xTrain = pd.DataFrame(data=xTrain, columns=datacols, dtype=float)
        xTrain['Behandlungskategorie'] = yTrain
        xVal = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
        xVal['Behandlungskategorie'] = yVal
        crossValSets.append([xTrain, xVal])
    return crossValSets, test


def splitFrameTestRest(dataset):
    data = dataset.drop(columns=['Behandlungskategorie'])
    datacols = list(data.columns)
    label = dataset['Behandlungskategorie']
    xRest, xVal, yRest, yVal = train_test_split(data.values, label.values,
                                                    random_state=42, test_size=0.33,
                                                    stratify=label)
    scaler = StandardScaler()
    scaler.fit(xRest)
    xRest = scaler.transform(xRest)
    xVal = scaler.transform(xVal)
    test = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
    test['Behandlungskategorie'] = yVal
    rest = pd.DataFrame(data=xRest, columns=datacols, dtype=float)
    rest['Behandlungskategorie'] = yRest
    return [test, rest]


def splitRestToTrainVal(dataset):
    data = dataset.drop(columns=['Behandlungskategorie'])
    datacols = list(data.columns)
    label = dataset['Behandlungskategorie']
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    crossValSets = []
    for i in range(0, 4):
        xTrain, xVal, yTrain, yVal = train_test_split(data, label.values, test_size=0.25, stratify=label.values, random_state=42)
        xTrain = pd.DataFrame(data=xTrain, columns=datacols, dtype=float)
        xTrain['Behandlungskategorie'] = yTrain
        xVal = pd.DataFrame(data=xVal, columns=datacols, dtype=float)
        xVal['Behandlungskategorie'] = yVal
        crossValSets.append([xTrain, xVal])
    return crossValSets



def scaleFrame(trainFrame, valFrame, testFrame):
    """
    Scales the datasets according to best practice in machine learning
    :param trainFrame: DataFrame containing training data
    :param valFrame: DataFrame containing validation data
    :param testFrame: DataFrame containing test data
    :return: Tuple of scaled datasets
    """
    scaler = StandardScaler()
    scaler.fit(trainFrame.drop(columns=['Behandlungskategorie']))
    scaledTrain = pd.DataFrame(scaler.transform(trainFrame.drop(columns=['Behandlungskategorie'])),
                               columns=trainFrame.drop(columns=['Behandlungskategorie']).columns, dtype=float)
    scaledVal = pd.DataFrame(scaler.transform(valFrame.drop(columns=['Behandlungskategorie'])),
                             columns=valFrame.drop(columns=['Behandlungskategorie']).columns, dtype=float)
    scaledTest = pd.DataFrame(scaler.transform(testFrame.drop(columns=['Behandlungskategorie'])),
                              columns=testFrame.drop(columns=['Behandlungskategorie']).columns, dtype=float)
    scaledTrain['Behandlungskategorie'] = trainFrame['Behandlungskategorie']
    scaledVal['Behandlungskategorie'] = valFrame['Behandlungskategorie']
    scaledTest['Behandlungskategorie'] = testFrame['Behandlungskategorie']
    return scaledTrain, scaledVal, scaledTest

def balanceData(data):
    """
    Returns the balanced dataset
    :param data: DataFrame containing patients and flags
    :return: DataFrame balanced to treatment categories
    """
    labelsData = data['Behandlungskategorie'].values
    labels, count = np.unique(labelsData, return_counts=True)
    print('Labels: ' + str(labels) + 'Counts: ' + str(count))
    minimum = np.amin(count)
    rest = minimum % 2
    threshold = count[np.argmin(count)] - rest
    balancedSet = []
    for i in range(0, labels.shape[0]):
        currData = data[data['Behandlungskategorie'] == labels[i]]
        randomChoice = np.random.choice(np.arange(0, currData.shape[0]), threshold, replace=False)
        balancedSet.append(currData.values[randomChoice])
        print('Label: ' + str(labels[i]) + ' Number of Samples: ' + str(currData.values[randomChoice].shape[0]))
    balancedSet = np.concatenate(balancedSet, axis=0)
    balancedSet = pd.DataFrame(data=balancedSet, columns=list(data.columns))
    return balancedSet

def loadReplacedSet():
    """
    Loads and cleans the dataset and adds the custom scores
    :return: DataFrame
    """
    all = [
    'BHPROFIEL1', 'BHSTATUS1', 'BHSTOPREDEN1',
    'BHVERW1',
    'BHVERWLOC1',
    'BHVERWINT1',
    'LWKNIH0110',
    'LWKNIH0210',
    'LWKNIH0310',
    'LWKNIH0410',
    'LWKNIH0510',
    'LWKNIH0610',
    'LWKNIH0710',
    'LWKNIH0810',
    'LWKWKC0110',
    'LWKWKC01_3_310',
    'LWKWKC0210',
    'LWKWKC02_NL10',
    'LWKWKC02_NL110',
    'LWKWKC02_CHTR310',
    'LWKWKC02_RMA10',
    'LWKWCK02_RMA110',
    'LWKWCK02_RMA210',
    'LWKWKC02_RMA310',
    'LWKWKC02_INT10',
    'LWKWKC02_INT110',
    'LWKWKC02_INT210',
    'LWKWKC02_INT310',
    'LWKWKC02_PL10',
    'LWKWKC02_PL110',
    'LWKWKC02_PL210',
    'LWKWKC02_PL310',
    'LWKWKC02_PSY10',
    'LWKWKC02_PSY110',
    'LWKWKC02_PSY210',
    'LWKWKC02_PSY310',
    'LWKWKC02_AND10',
    'LWKWKC03_0110',
    'LWKWKC03_0210',
    'LWKWKC03_0310',
    'LWKWKC03_0410',
    'LWKWKC03_0510',
    'LWKWKC03_0610',
    'LWKWKC03_0710',
    'LWKWKC03_0810',
    'LWKWKC03_0910',
    'LWKWKC03_1010',
    'LWKWKC03_1110',
    'LWKWKC03_1210',
    'LWKWKC03_1310',
    'LWKWKC03_1410',
    'LWKWKC03_1510',
    'LWKWKC03_1610',
    'LWKWKC03_1710',
    'LWKWKC03_1810',
    'LWKWKC03_1910',
    'LWKWKC03_2010',
    'LWKWKC03_2110',
    'LWKWKC03_2210',
    'LWKWKC03_2310',
    'LWKWKC03_2410',
    'LWKNIH09_110',
    'LWKNIH09_210',
    'LWKNIH09_310',
    'LWKNIH09_410',
    'LWKNIH09_1_110',
    'LWKWKC0410',
    'LWKWKC04_110',
    'LWKWKC04_210',
    'LWKWKC04_310',
    'LWKWKC04_410',
    'LWKWKC04_3410',
    'LWKWKC04_510',
    'LWKWKC04_610',
    'LWKWKC04_6_110',
    'LWKNIH1010',
    'LWKNIH1110',
    'LWKNIH1210',
    'LWKNIH1310',
    'LWKNIH1410',
    'LWKNIH1510',
    'LWKNIH1610',
    'LWKNIH1710',
    'LWKNIH1810',
    'LWKNIH1910',
    'LWKNIH2010',
    'LWKNIH2110',
    'LWKNIH2210',
    'LWKNIH2310',
    'LWKNIH2410',
    'LWKNIH2510',
    'LWKNIH2610',
    'LWKNIH2710', 'LWKNIH2810', 'LWKNIH2910', 'LWKNIH3010',
    'LWKNIH3110', 'LWKNIH3210', 'LWKNIH3310', 'LWKNIH3410', 'LWKNIH3510', 'LWKNIH3610', 'LWKNIH3710', 'LWKNIH37_110', \
    'LWKNIH3810', 'LWKNIH39_0110', 'LWKNIH39_0210', 'LWKNIH39_0310', 'LWKNIH39_0410', 'LWKNIH39_0510', 'LWKNIH39_0610',
    'LWKNIH39_0710', 'LWKNIH39_0810', 'LWKNIH39_0910', 'LWKNIH39_1010', 'LWKNIH39_1110', 'LWKNIH39_1210', 'LWKNIH39_11_110', 'LWKNIH4010', 'LWKNIH4110']
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
             'GSCPCQ0710', 'GSCPCQ0810', 'GSCPCQ0910', 'GSCPCQ1010']

    undetermined = ['LWKWKC03_1410', 'LWKWKC03_2210', 'LWKWKC03_2310',
                    'LWKWKC03_1310', 'LWKNIH09_310', 'LWKWKC03_1210',
                    'LWKNIH09_410', 'LWKWKC03_2410']

    misc = ['GESLACHT']
    customScores = ['Stress_arbeit_privatleben', 'alcohol_drugs', 'Depression_Index', 'Schlafqualität']
    flags = red + yellow + blue + black + misc + undetermined + all + customScores
    flags = list(dict.fromkeys(flags))
    umcgPath = './Data/umcg_treat.csv'
    cleaned = cleanData(umcgPath)
    cleaned = dimensionalityReduction(cleaned)
    doi = cleaned[flags + ['Behandlungskategorie']]
    doi['Behandlungskategorie'] = pd.to_numeric(doi['Behandlungskategorie'])
    print('DOI shape: ' + str(doi.shape))
    treatment = doi[doi['Behandlungskategorie'].notna()]
    print('Treatment shape: ' + str(treatment.shape))
    treatment = fillOmittedValues(treatment)
    print('Filled shape: ' + str(treatment.shape))
    print('Reduced shape: ' + str(treatment.shape))
    treatment.dropna(axis=1, inplace=True)
    treatment.dropna(axis=0, inplace=True)
    for colname, colvalues in treatment.iteritems():
        try:
            pd.to_numeric(colvalues)
        except:
            treatment.drop(columns=[colname], inplace=True)
    print('Dropped shape: ' + str(treatment.shape))
    unique, counts = np.unique(treatment.values, return_counts=True, axis=1)
    print('Different patients: ' + str(len(unique)))
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] != 3] = 0
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] > 0] = 1
    return treatment

def loadAsFrame(flags):
    """
    Generates a training, validation and test set for the given items/flags
    :param flags: list of strings of items
    :return: list of DataFrames
    """
    umcgPath = './Data/umcg_treat.csv'
    cleaned = cleanData(umcgPath)
    flags.append('Behandlungskategorie')
    doi = cleaned[flags]
    doi['Behandlungskategorie'] = pd.to_numeric(doi['Behandlungskategorie'])
    print('DOI shape: ' + str(doi.shape))
    treatment = doi[doi['Behandlungskategorie'].notna()]
    print('Treatment shape: ' + str(treatment.shape))
    treatment = fillOmittedValues(treatment)
    print('Filled shape: ' + str(treatment.shape))
    treatment = dimensionalityReduction(treatment)
    print('Reduced shape: ' + str(treatment.shape))
    treatment.dropna(axis=0, inplace=True)
    print('Dropped shape: ' + str(treatment.shape))
    treatment = treatment[np.logical_or(treatment['Behandlungskategorie'] == 6, treatment['Behandlungskategorie'] < 3)]
    treatment[treatment['Behandlungskategorie'] < 3] = treatment[treatment['Behandlungskategorie'] < 3] - 1
    treatment[treatment['Behandlungskategorie'] == 6] = 2
    print(np.unique(treatment['Behandlungskategorie'], return_counts=True))
    train, val, test = splitFrame(treatment)
    xtrain, xval, xtest = scaleFrame(train, val, test)
    return xtrain, xval, xtest

def computeMetric(dataset):
    """
    Computes the pairwise distance for each patient in the dataset
    :param dataset: ndarray containing the flags
    :return: ndarray of pairwise distances
    """
    data = dataset.values
    dists = []
    for i in range(0, data.shape[0]):
        for j in range(i+1, data.shape[0]):
            similarities = np.logical_not(np.equal(data[i, :], data[j, :]))
            distance = np.sum(similarities)
            dists.append(distance)
    return np.asarray(dists)

def prepareForComparison(dataframe):
    """
    Rescales the values to make patients more equal
    :param dataframe: DataFrame with patient flags
    :return: DataFrame
    """
    dropCols = ['GSCCOPSOQ0110', 'GSCCOPSOQ0210', 'GSCCOPSOQ0310', 'GSCCOPSOQ0410', 'GSCCOPSOQ0510', 'GSCCOPSOQ0610',
                'GSCCOPSOQ0710', 'GSCCOPSOQ0810', 'GSCCOPSOQ0910', 'GSCCOPSOQ1010', 'GSCCOPSOQ1110', 'GSCCOPSOQ1410',
                'GSCCOPSOQ1510', 'GSCCOPSOQ1710', 'GSCCOPSOQ1810', 'GSCCOPSOQ1910', 'GSCCOPSOQ2010', 'GSCCOPSOQ2410',
                'GSCCOPSOQ2510', 'LWKNIH09_310', 'LWKNIH09_410', 'BHPROFIEL1', 'BHSTATUS1']
    dataframe.drop(columns=dropCols, inplace=True)
    print('New shape: ' + str(dataframe.shape))
    # 1 and 2, same; > 0 = 1
    c1 = ['LWKNIH1110', 'LWKNIH0610']
    print(str(c1) + ': ' + str(np.unique(dataframe[c1].values)))
    for col in c1:
        dataframe.loc[dataframe[col] > 0, col] = 1
    print(str(c1) + ': ' + str(np.unique(dataframe[c1].values)))
    # 0 and 1,2,3 same and 4; >0 and <4 = 1
    c2 = ['LWKNIH2210', 'LWKNIH2310', 'LWKNIH2410', 'LWKNIH2510', 'LWKNIH2710', 'LWKNIH2810', 'LWKNIH2910']
    print(str(c2) + ': ' + str(np.unique(dataframe[c2].values)))
    for col in c2:
        dataframe.loc[(dataframe[col] > 0) & (dataframe[col] < 4), col] = 1
    print(str(c2) + ': ' + str(np.unique(dataframe[c2].values)))
    # <4 = 0; >=4 = 1
    c3 = 'LWKNIH2610'
    print(str(c3) + ': ' + str(np.unique(dataframe[c3].values)))
    dataframe.loc[dataframe[c3] < 4, c3] = 0
    dataframe.loc[dataframe[c3] >= 4, c3] = 1
    print(str(c3) + ': ' + str(np.unique(dataframe[c3].values)))
    # <2 = 0 >2=1
    c4 = ['LWKNIH3210', 'LWKNIH3310', 'LWKNIH1410', 'LWKNIH1510', 'LWKNIH1610', 'LWKNIH1710',
          'GSCCOPSOQ1210', 'GSCCOPSOQ1310', 'GSCCOPSOQ1610', 'GSCCOPSOQ2110']
    print(str(c4) + ': ' + str(np.unique(dataframe[c4].values)))
    for col in c4:
        dataframe.loc[(dataframe[col] < 2), col] = 0
        dataframe.loc[(dataframe[col] > 2), col] = 1
    print(str(c4) + ': ' + str(np.unique(dataframe[c4].values)))
    # 3>x>0 = 1
    c7 = ['GSCCOPSOQ2210', 'GSCCOPSOQ2310']
    print(str(c7) + ': ' + str(np.unique(dataframe[c7].values)))
    for col in c7:
        dataframe.loc[(dataframe[col] > 0) & (dataframe[col] < 3), col] = 1
    print(str(c7) + ': ' + str(np.unique(dataframe[c7].values)))
    # <=50 =0, >50 <100 = 1, >= 100 = 2
    c8 = ['GSCPCQ01_110', 'LWKNIH3610']
    print(str(c8) + ': ' + str(np.unique(dataframe[c8].values)))
    for col in c8:
        dataframe.loc[(dataframe[col] <= 50), col] = 0
        dataframe.loc[(dataframe[col] > 50) & (dataframe[col] < 100), col] = 1
        dataframe.loc[(dataframe[col] >= 100), col] = 2
    print(str(c8) + ': ' + str(np.unique(dataframe[c8].values)))
    # <3 = 0, 6>x>=3 = 1, >=6 = 2
    c9 = 'GSCPCQ0210'
    print(str(c9) + ': ' + str(np.unique(dataframe[c9].values)))
    dataframe.loc[(dataframe[c9] < 3), c9] = 0
    dataframe.loc[(dataframe[c9] >= 3) & (dataframe[c9] < 6), c9] = 3
    dataframe.loc[(dataframe[c9] >= 6), c9] = 2
    print(str(c9) + ': ' + str(np.unique(dataframe[c9].values)))
    # <11 = 0, 20>=x>=11 = 1, >20 = 2
    c10 = ['GSCPCQ03_110', 'GSCPCQ0710', 'GSCPCQ1010']
    print(str(c10) + ': ' + str(np.unique(dataframe[c10].values)))
    for col in c10:
        dataframe.loc[(dataframe[col] < 11), col] = 0
        dataframe.loc[(dataframe[col] >= 11) & (dataframe[col] <= 20), col] = 1
        dataframe.loc[(dataframe[col] > 20), col] = 2
    print(str(c10) + ': ' + str(np.unique(dataframe[c10].values)))
    # <4 = 0, 6>=x>=4 = 1, >6 = 2
    c11 = ['GSCPCQ0810', 'LWKNIH0310', 'LWKNIH0510']
    print(str(c11) + ': ' + str(np.unique(dataframe[c11].values)))
    for col in c11:
        dataframe.loc[(dataframe[col] < 4), col] = 0
        dataframe.loc[(dataframe[col] >= 4) & (dataframe[col] <= 6), col] = 1
        dataframe.loc[(dataframe[col] > 6), col] = 2
    print(str(c11) + ': ' + str(np.unique(dataframe[c11].values)))
    # <5 = 0
    c12 = 'LWKNIH0110'
    print(str(c12) + ': ' + str(np.unique(dataframe[c12].values)))
    dataframe.loc[dataframe[c12] < 5, c12] = 0
    print(str(c12) + ': ' + str(np.unique(dataframe[c12].values)))
    # 3>x>0 = 1, >2 = 3
    c13 = ['LWKNIH1810', 'LWKNIH1910', 'LWKNIH2010', 'LWKNIH2110']
    print(str(c13) + ': ' + str(np.unique(dataframe[c13].values)))
    for col in c13:
        dataframe.loc[(dataframe[col] > 2), col] = 3
        dataframe.loc[(dataframe[col] < 3) & (dataframe[col] > 0), col] = 1
    print(str(c13) + ': ' + str(np.unique(dataframe[c13].values)))
    # =2 ->0
    c14 = 'LWKNIH3410'
    print(str(c14) + ': ' + str(np.unique(dataframe[c14].values)))
    dataframe.loc[dataframe[c14] == 2, c14] = 0
    print(str(c14) + ': ' + str(np.unique(dataframe[c14].values)))
    # <=160 = 1, 190>=x>160 =2, >190 = 3
    c15 = 'LWKNIH3510'
    print(str(c15) + ': ' + str(np.unique(dataframe[c15].values)))
    dataframe.loc[(dataframe[c15] <= 160), c15] = 1
    dataframe.loc[(dataframe[c15] > 190), c15] = 3
    dataframe.loc[(dataframe[c15] > 160) & (dataframe[c15] <= 190), c15] = 2
    print(str(c15) + ': ' + str(np.unique(dataframe[c15].values)))
    # <3 = 1, 7>=x>=3 =2
    c16 = 'LWKNIH3710'
    print(str(c16) + ': ' + str(np.unique(dataframe[c16].values)))
    dataframe.loc[(dataframe[c16] < 3), c16] = 1
    dataframe.loc[(dataframe[c16] <= 7) & (dataframe[c16] >= 3), c16] = 2
    print(str(c16) + ': ' + str(np.unique(dataframe[c16].values)))
    return dataframe

def findToThreshold(dataframe, threshold):
    """
    Computes finds similar patients to a given threshold
    :param dataframe: DataFrame containing the patient data
    :param threshold: int threshold for considering two patients as similar
    :return: DataFrame containing all similar patients to the threshold value
    """
    patients = []
    for i in range(0, dataframe.shape[0]):
        for j in range(i + 1, dataframe.shape[0]):
            similarities = np.logical_not(np.equal(dataframe.iloc[i, :-1].values, dataframe.iloc[j, :-1].values))
            distance = np.sum(similarities)
            if distance <= threshold:
                patients.append(dataframe.iloc[i, :].values.reshape(1, -1))
                patients.append(dataframe.iloc[j, :].values.reshape(1, -1))
    patients = np.concatenate(patients, axis=0)
    patientFrame = pd.DataFrame(data=patients, columns=dataframe.columns)
    patientFrame = patientFrame.drop_duplicates()
    return patientFrame

def getTreatmentPaths(patientFrame, pathCounts):
    """
    Computes and prints the number of patients for each treatment paths
    :param patientFrame: DataFrame containing the items
    :param pathCounts: ndarray containing the counts of treatmentpaths
    :return: None
    """
    treat, counts = np.unique(patientFrame['Behandlungskategorie'].values, return_counts=True)
    print('Paths: ' + str(treat) + ' Counts: ' + str(counts))
    for t in range(0, len(treat)):
        print('Absolute number of patients for treatmentpath ' + str(treat[t]) + ' is ' + str(counts[t]))
        print('Relative number of patients for treatmentpath ' + str(treat[t]) + ' is ' +
              str(np.divide(counts[t], pathCounts[int(treat[t]-1)])))
    return

def loadForSil():
    """
    Loads the crossval dataset for Silhouette analyses
    :return: DataFrame
    """
    data = loadReplacedSetCrossval()
    adjusted = prepareForComparison(data.drop(columns=['Behandlungskategorie']))
    adjustedT = pd.concat([adjusted, data['Behandlungskategorie']], axis=1)
    return adjustedT

def computeMissingPercent():
    """
    Computes the percentage of missing values of the list of red/yellow/blue/black flags
    :return: ndarray of percentages
    """
    listofitems = ['LWKNIH0110', 'LWKNIH0210', 'LWKNIH0310', 'LWKNIH0410', 'LWKNIH0510', 'LWKNIH0610',
                   'LWKNIH0710', 'LWKNIH0810', 'LWKWKC0110', 'LWKWKC01_1_110', 'LWKWKC01_1_210',
                   'LWKWKC01_1_310', 'LWKWKC01_2_110', 'LWKWKC01_2_210', 'LWKWKC01_2_310',
                   'LWKWKC01_3_110', 'LWKWKC01_3_210', 'LWKWKC01_3_310', 'LWKWKC01_4_110',
                   'LWKWKC01_4_210', 'LWKWKC01_4_310', 'LWKWKC01_5_110', 'LWKWKC01_5_210',
                   'LWKWKC01_5_310', 'LWKWKC0210', 'LWKWKC02_NL10', 'LWKWKC02_NL110',
                   'LWKWKC02_NL210', 'LWKWKC02_NL310', 'LWKWKC02_REV10', 'LWKWKC02_REV110', 'LWKWKC02_REV210',
                   'LWKWKC02_REV310', 'LWKWKC02_PIJN10', 'LWKWKC02_PIJN110', 'LWKWKC02_PIJN210', 'LWKWKC02_PIJN310',
                   'LWKWKC02_NC10', 'LWKWKC02_NC110', 'LWKWKC02_NC210', 'LWKWKC02_NC310', 'LWKWKC02_ORT10',
                   'LWKWKC02_ORT110', 'LWKWKC02_ORT210', 'LWKWKC02_ORT310', 'LWKWKC02_CHTR10', 'LWKWKC02_CHTR110',
                   'LWKWKC02_CHTR210', 'LWKWKC02_CHTR310', 'LWKWKC02_RMA10', 'LWKWCK02_RMA110', 'LWKWCK02_RMA210',
                   'LWKWKC02_RMA310', 'LWKWKC02_INT10', 'LWKWKC02_INT110', 'LWKWKC02_INT210', 'LWKWKC02_INT310',
                   'LWKWKC02_PL10', 'LWKWKC02_PL110', 'LWKWKC02_PL210', 'LWKWKC02_PL310', 'LWKWKC02_PSY10',
                   'LWKWKC02_PSY110', 'LWKWKC02_PSY210', 'LWKWKC02_PSY310', 'LWKWKC02_AND10', 'LWKWKC02_AND010',
                   'LWKWKC02_AND110', 'LWKWKC02_AND210', 'LWKWKC02_AND310', 'LWKWKC03_0110', 'LWKWKC03_0210',
                   'LWKWKC03_0310', 'LWKWKC03_0410', 'LWKWKC03_0510', 'LWKWKC03_0610', 'LWKWKC03_0710',
                   'LWKWKC03_0810', 'LWKWKC03_0910', 'LWKWKC03_1010', 'LWKWKC03_1110', 'LWKWKC03_1210',
                   'LWKWKC03_1310', 'LWKWKC03_1410', 'LWKWKC03_1510', 'LWKWKC03_1610', 'LWKWKC03_1710',
                   'LWKWKC03_1810', 'LWKWKC03_1910', 'LWKWKC03_2010', 'LWKWKC03_2110', 'LWKWKC03_2210',
                   'LWKWKC03_2310', 'LWKWKC03_2410', 'LWKNIH09_110', 'LWKNIH09_210', 'LWKNIH09_310', 'LWKNIH09_410',
                   'LWKNIH09_1_110', 'LWKWKC0410', 'LWKWKC04_110', 'LWKWKC04_210', 'LWKWKC04_310', 'LWKWKC04_410',
                   'LWKWKC04_3410', 'LWKWKC04_510', 'LWKWKC04_610', 'LWKWKC04_6_110', 'LWKNIH1010', 'LWKNIH1110', 'LWKNIH1210',
                   'LWKNIH1310', 'LWKNIH1410', 'LWKNIH1510', 'LWKNIH1610', 'LWKNIH1710', 'LWKNIH1810', 'LWKNIH1910',
                   'LWKNIH2010', 'LWKNIH2110', 'LWKNIH2210', 'LWKNIH2310', 'LWKNIH2410', 'LWKNIH2510', 'LWKNIH2610',
                   'LWKNIH2710', 'LWKNIH2810', 'LWKNIH2910', 'LWKNIH3010', 'LWKNIH3110', 'LWKNIH3210', 'LWKNIH3310',
                   'LWKNIH3410', 'LWKNIH3510', 'LWKNIH3610', 'LWKNIH3710', 'LWKNIH37_110', 'LWKNIH3810', 'LWKNIH39_0110',
                   'LWKNIH39_0210', 'LWKNIH39_0310', 'LWKNIH39_0410', 'LWKNIH39_0510', 'LWKNIH39_0610', 'LWKNIH39_0710',
                   'LWKNIH39_0810', 'LWKNIH39_0910', 'LWKNIH39_1010', 'LWKNIH39_1110', 'LWKNIH39_1210', 'LWKNIH39_11_110',
                   'LWKNIH4010', 'LWKNIH4110']
    dataset = pd.read_csv('./Data/GSC_dataset_anonymous.csv', sep=';', low_memory=False)
    subFrame = dataset[dataset['Treatmentcategory'] == '3']
    subFrame = subFrame.replace(r'^\s*$', np.nan, regex=True)
    missing = subFrame[[c for c in subFrame.columns if c in listofitems]].isna().sum()
    percent = np.divide(missing.values, subFrame.shape[0])
    return percent

def plotPCAVariance():
    """
    Computes and plots the number of principle components dependent on the variance to keep
    :return: None
    """
    data = loadReplacedSet()
    data = data.drop(columns=['Behandlungskategorie']).values
    n_components = []
    variance = np.concatenate([np.arange(0.1, 1, 0.1), [0.99]])
    for i in range(0, len(variance)):
        pca = PCA(variance[i])
        pca.fit(data)
        n_components.append(pca.n_components_)
    plt.plot(variance, n_components)
    plt.xlabel('Retained Variance in the Data')
    plt.ylabel('Number of Items')
    plt.show()
    return