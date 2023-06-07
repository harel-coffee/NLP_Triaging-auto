"""
This file contains an EA using kNN as fitness function
"""
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.spatial import distance
import joblib
import sys
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

# get a list of all flags
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
         'GSCPCQ0710', 'GSCPCQ0810', 'GSCPCQ0910', 'GSCPCQ1010']  # 'GSCPCQ0510', rausgenommen weil timestamp

undetermined = ['LWKWKC03_1410', 'LWKWKC03_2210', 'LWKWKC03_2310',
                'LWKWKC03_1310', 'LWKNIH09_310', 'LWKWKC03_1210',
                'LWKNIH09_410', 'LWKWKC03_2410']
customScores = ['Stress_arbeit_privatleben', 'alcohol_drugs', 'Depression_Index', 'Schlafqualität']

global flag
flags = red + yellow + blue + black + undetermined + customScores

def customMetric(train, val, weights):
    """
    Pre-computes the custom metric for the kNN classifier, needed for using weights
    :param train: ndarray containing the training data and the label in the last column
    :param val: ndarray containing the validation data and the label in the last column
    :param weights: ndarray of weights for each label
    :return: ndarray of pairwise distances
    """
    dist_matrix = np.zeros(shape=(train.shape[0], val.shape[0]))
    for i in range(0, train.shape[0]):
        for j in range(0, val.shape[0]):
            dist = distance.euclidean(train[i, :], val[j, :])
            boolArray = (weights[:, 0:-1] == val[j, :])
            w_index = np.where(boolArray.all(axis=1))[0]
            try:
                w = int(w_index)
            except:
                w = int(w_index[0])
            dist = np.multiply(dist, weights[w, -1])
            dist_matrix[i, j] = dist
    return dist_matrix

def decode(indices):
    """
    Decodes the individual to get the items
    :param indices: ndarray of ints
    :return: list of items
    """
    decoded = [flags[j] for j in indices]
    return decoded

def fitnessFunction(trainingData, validationData, individuals):
    """
    The fitness function using the standard training and validation split
    The last population is discarded
    :param trainingData: ndarray containing the training data and the labels in the last column
    :param validationData: ndarray containing the validation data and the labels in the last column
    :param individuals: list of ndarrays of individuals
    :return: best n individuals
    """
    bestPop = []
    for individual in individuals:
        if len(individual) == 3:
            bestPop.append(individual)
            continue
        items = individual
        bestScore = 0
        bestCLF = None
        for j in range(1, 100):
            clf = KNeighborsClassifier(n_neighbors=j, n_jobs=-1, weights='distance')
            clf.fit(trainingData[:, items], trainingData[:, -1])
            y_pred = clf.predict(validationData[:, items])
            score = f1_score(validationData[:, -1], y_pred, average='macro')
            if score > bestScore:
                bestScore = score
                bestCLF = clf
        bestPop.append([individual, bestScore, bestCLF])
    bestPop = sorted(bestPop, key=lambda x: x[1], reverse=True)
    return bestPop

def fitnessFunctionCV(dataset, individuals, preWeights):
    """
    The fitness function using cross validation for the kNN classifier
    :param dataset: ndarray of patients and items
    :param individuals: list of ndarrays containing the current generation
    :param preWeights: ndarray of precomputed weights for the custom metric
    :return: the best n individuals
    """
    bestPop = []
    xTrain = dataset[:, 0:-1]
    yTrain = dataset[:, -1]
    crossValPairs = []
    for i in range(0, 4):
        tempTrain, tempVal, tempTrainLabel, tempValLabel = train_test_split(xTrain, yTrain,
                                                                            test_size=0.15,
                                                                            stratify=yTrain)
        crossValPairs.append([tempTrain, tempTrainLabel,
                              tempVal, tempValLabel])
    for individual in individuals:
        try:
            if isinstance(individual[2], dict):
                bestPop.append(individual)
                continue
        except Exception as exc:
            exc = None
        items = individual
        for index, cross in enumerate(crossValPairs):
            dist_matrix_train = customMetric(cross[0][:, items], cross[0][:, items], preWeights[:, np.append(items, -1)])
            dist_matrix_val = customMetric(cross[2][:, items], cross[0][:, items], preWeights[:, np.append(items, -1)])
            crossValPairs[index].append(dist_matrix_train)
            crossValPairs[index].append(dist_matrix_val)
        bestScore = 0
        bestCLF = None
        for j in range(1, 10):
            crossDict = {'estimator': [], 'test_score': []}
            for crossval in crossValPairs:
                clf = KNeighborsClassifier(n_neighbors=j, n_jobs=8,
                                           weights='distance',
                                           metric='precomputed')
                clf.fit(crossval[4], crossval[1])
                y_pred = clf.predict(crossval[5])
                fscore = f1_score(crossval[3], y_pred, average='macro')
                crossDict['estimator'].append(clf)
                crossDict['test_score'].append(fscore)
            if np.amax(crossDict['test_score']) > bestScore:
                bestScore = np.amax(crossDict['test_score'])
                crossDict['data'] = crossValPairs
                bestCLF = crossDict
        bestPop.append([individual, bestScore, bestCLF])
    bestPop = sorted(bestPop, key=lambda x: x[1], reverse=True)
    return bestPop

def recombine(listOfIndividuals):
    """
    Recombines the current generation for the EA recombination step
    :param listOfIndividuals: list of ndarrays containing the current generation
    :return: list of ndarrays contining the offspring
    """
    offspring = []
    for k in range(0, len(listOfIndividuals), 2):
        parent1 = listOfIndividuals[k][0]
        parent2 = listOfIndividuals[k+1][0]
        try:
            lenP1 = int(np.divide(int(len(parent1)), 2))
        except Exception as exc:
            parent1 = np.asarray([parent1])
            lenP1 = int(np.divide(int(len(parent1)), 2))
        try:
            lenP2 = int(np.divide(int(len(parent2)), 2))
        except Exception as exc:
            parent2 = np.asarray([parent2])
            lenP2 = int(np.divide(int(len(parent2)), 2))
        child = np.hstack((parent1[0:lenP1], parent2[lenP2:]))
        if len(child) > 87:
            print('Child too long: ' + str(len(child)))
        elif len(child) == 0:
            print('Child too short: ' + str(len(child)))
        offspring.append(child)
    return offspring

def mutate(listOfIndividuals):
    """
    Mutates the individuals for the EA mutation step
    :param listOfIndividuals: ndarrays containing individuals
    :return: list of ndarrays containing the offspring
    """
    offspring = []
    for ind in listOfIndividuals:
    # first mutate length
        length_add = np.random.randint(-1, 2, size=1)
        if 88 > len(ind)+length_add > 0:
            if len(ind)+length_add > len(ind):
                newItems = np.random.randint(0, 87, size=length_add)
                new = np.hstack((newItems, ind))
            elif len(ind)+length_add < len(ind):
                randomChoice = np.random.randint(0, len(ind), size=len(ind)+length_add)
                new = ind[randomChoice]
            else:
                new = ind
        elif len(ind)+length_add < 0:
            new = np.random.randint(0, 87, size=44)
        else:
            new = ind
        #second mutate items by mutating the indices
        alterItems = np.random.randint(-3, 4, size=len(new))
        new = new + alterItems
        new[new < 0] = 0
        new[new > 86] = 86
        offspring.append(new)
    return offspring

def doItRight(data, weights, name):
    """
    Computes the evolutionary algorithm with kNN classifiers as fitness function
    Prints the parameters for the best run
    :param data: ndarray containing the dataset and labels in the last column
    :param weights: ndarray containing the weights for each label
    :param name: string name of the process
    :return: None
    """
    savePath = './models/knn/'
    processName = name
    # 1. initialise generation
    population = []
    m = 20
    l = 10
    for i in range(0, m):
        randSize = np.random.randint(1, 87)
        population.append(np.random.randint(0, 87, size=randSize))
    # for x generations
    generations = 10
    bestIndividual = [0, 0, None]
    earlyStopping = 0
    weights = np.concatenate([data[:, 0:-1], weights], axis=1)
    for g in range(0, generations):
        print('Generation: ' + str(g) + ' of process ' + str(processName))
        # 2. evaluate fitness
        fitPop = fitnessFunctionCV(data, population, weights)
        print('Best specimen score: ' + str(fitPop[0][1]) + ' of process ' + str(processName))
        # 3. select fittest
        if bestIndividual[1] < fitPop[0][1]:
            bestIndividual = fitPop[0]
            earlyStopping = 0
        else:
            earlyStopping = earlyStopping + 1
            if earlyStopping > 100:
                print('Early stopping of ' + str(processName))
                break
        # 4. recombine and mutate them
        recombined = recombine(fitPop[0:l])
        offspring = mutate(recombined)
        parents = fitPop[0:m]
        population.clear()
        # 5. strategy dependent: keep or omit
        population = offspring + parents
    try:
        print('Best specimen of process ' + str(processName) + ' has ' + str(len(bestIndividual[0])) +
              ' items with a ')
        print("F1-Score: %0.2f (+/- %0.2f)" % (
        np.asarray(bestIndividual[2]['test_score']).mean(), np.asarray(bestIndividual[2]['test_score']).std() ** 2))
    except Exception as exc:
        print('Best specimen of process ' + str(processName) + ' has ' + str(np.array(bestIndividual[0]).shape) +
              ' items with a ')
        print("F1-Score: %0.2f (+/- %0.2f)" % (
        np.asarray(bestIndividual[2]['test_score']).mean(), np.asarray(bestIndividual[2]['test_score']).std() ** 2))
    return

def computeWeights(data):
    """
    Computes the weights for each label
    :param data: ndarray containing the labels in the last column
    :return: ndarray containing the weights for each class
    """
    labels, counts = np.unique(data[:, -1], return_counts=True)
    weights = np.zeros(shape=(data.shape[0], 1))
    for i in range(0, labels.shape[0]):
        w = np.multiply(np.divide(1, counts[i]), np.divide(data.shape[0], 2))
        indices = np.where(data[:, -1] == labels[i])[0]
        weights[indices] = w
    return weights

def example(param):
    """
    Runs an EA example
    :return:
    """
    basePath = './Data/dataset/'
    train = np.load(basePath + 'training_balanced.npy')
    val = np.load(basePath + 'validation_balanced.npy')
    test = np.load(basePath + 'test_balanced.npy')
    data = np.concatenate([train, val], axis=0)
    weights = computeWeights(data)
    #Add custom flags
    #EA m=1000 + l=500 1000 fittest for the next generation
    doItRight(data, weights, param)
    return

#for local use
# if __name__ == '__main__':
#     param = sys.argv[1]
#     #load data
#     #train, val, test = dt.loadAsFrame(flags)
#     basePath = './Data/dataset/'
#     #train = pd.read_csv(basePath + 'training.csv', sep=';')
#     #val = pd.read_csv(basePath + 'validation.csv', sep=';')
#     #test = pd.read_csv(basePath + 'test.csv', sep=';')
#     train = np.load(basePath + 'training.npy')
#     val = np.load(basePath + 'validation.npy')
#     test = np.load(basePath + 'test.npy')
#     data = np.concatenate([train, val], axis=0)
#     #doTheStuff(train, val, param)
#     #Add custom flags
#     #EA m=1000 + l=500 1000 fittest for the next generation
#     print('main line')
#     procs = []
#     for i in range(0, 24):
#         p = Process(target=doItRight, args=(data, test, param), name=str(i))
#         p.start()
#     for proc in procs:
#         proc.join()
example(1)