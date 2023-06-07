from skrebate import ReliefF
import pandas as pd

all = pd.read_csv("../Data/verf_neu/oold/merged_nocustom/multi/raw_data.csv", sep=';', low_memory=False, decimal=',')
# all.sample(frac=1)
all.drop(all.columns[0], axis=1, inplace=True)

target = all.pop("Behandlungskategorie")

re = ReliefF(n_neighbors=50).fit(all.values, target)
for ind, val in enumerate(re.feature_importances_):
    print(str(all.columns[ind]) + ": " + str(val))

pd.array()