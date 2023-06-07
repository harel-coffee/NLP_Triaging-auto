import numpy as np
from graphviz import Source
from sklearn import tree
import pandas as pd


def loadData(pathToCrossValData):
    """
    Loads the data splitted into four cross validation sets
    :param pathToCrossValData: string
    :return: list of data using all features, list of data using features identified as flags only
    """
    #   dataFlags = []
    #   labelsFlags = []
    dataAll = []
    labelsAll = []
    valAllData = np.load(pathToCrossValData + 'validation_all_data.npy')
    valAllLabel = np.load(pathToCrossValData + 'validation_all_label.npy')
    #    valFlagsData = np.load(pathToCrossValData + 'validation_flags_data.npy')
    #    valFlagsLabel = np.load(pathToCrossValData + 'validation_flags_label.npy')
    for i in range(0, 4):
        currentAllDataPath = pathToCrossValData + 'fold_' + str(i) + '_val_data.npy'
        currentAllLabelPath = pathToCrossValData + 'fold_' + str(i) + '_val_label.npy'
        #        currentFlagsDataPath = pathToCrossValData + 'fold_' + str(i) + '_flags_data.npy'
        #        currentFlagsLabelPath = pathToCrossValData + 'fold_' + str(i) + '_flags_label.npy'
        #        dataFlags.append(np.load(currentFlagsDataPath))
        #        labelsFlags.append(np.load(currentFlagsLabelPath))
        dataAll.append(np.load(currentAllDataPath))
        labelsAll.append(np.load(currentAllLabelPath))
    return [dataAll, labelsAll, valAllData, valAllLabel]


# all = loadData("./Data/verf_neu/merged/" + "1/")
all = pd.read_csv("Data/verf_neu/oold/merged_noextra_top50/multi/raw_data.csv", sep=';', low_memory=False, decimal=',')
all.sample(frac=1)
all.drop(all.columns[0], axis=1, inplace=True)

target = all.pop("Behandlungskategorie")
clf = tree.DecisionTreeClassifier()

clf.fit(all, target)

tree.plot_tree(clf)
print("DONE")

graph = Source(tree.export_graphviz(clf, out_file=None, feature_names=all.columns))
graph.format = 'png'
graph.render('dtree_render', view=True)
