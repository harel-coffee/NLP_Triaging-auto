import pandas as pd
from skfeature.function.similarity_based import fisher_score

all = pd.read_csv("../Data/verf_neu/oold/merged_nocustom/multi/raw_data.csv", sep=';', low_memory=False, decimal=',')
#all.sample(frac=1)
all.drop(all.columns[0], axis=1, inplace=True)

target = all.pop("Behandlungskategorie")

idx = fisher_score.fisher_score(all.values, target,
                                mode='rank')  # returns rank directly instead of fisher score. so no need for feature_ranking
for ind, val in enumerate(idx):
    print(str(all.columns[ind]) + ": " + str(val))
