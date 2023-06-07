import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from skrebate import ReliefF

from param_analyse import fisher_score_custom


def loadCsv(from_name):
    data = pd.read_csv(from_name, sep=';', low_memory=False, decimal=',')
    # all.sample(frac=1)
    data.drop(data.columns[0], axis=1, inplace=True)
    target = data.pop("Behandlungskategorie")

    scaler = StandardScaler()

    # create copy of DataFrame

    # created scaled version of DataFrame
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    return data, target

def loadTranslation():
    data = pd.read_csv("../Data/umcg_translation.csv", sep=";")
    data.drop(columns=["dutch"], inplace=True)
    return data


def fisher(x, y):
    arr = []
    idx = fisher_score.fisher_score(x.values, y,
                                    mode='index')  # returns rank directly instead of fisher score. so no need for feature_ranking
    for ind, val in enumerate(idx):
        arr.append([x.columns[val], max(idx) - ind])
        # print(str(x.columns[ind]) + ": " + str(val))
    return pd.DataFrame(arr, columns=["Column", "Fisher"])


def fisher_sc(x, y, name=""):
    arr = []
    idx = fisher_score_custom.fisher_score(x.values, y)
    for ind, val in enumerate(idx):
        arr.append([x.columns[ind], val])
        # print(str(x.columns[ind]) + ": " + str(val))
    return pd.DataFrame(arr, columns=["Column", "Fisher_score" + name])


def relieff(x, y, name=""):
    arr = []
    re = ReliefF(n_neighbors=50, n_jobs=-1).fit(x.values, y)
    for ind, val in enumerate(re.feature_importances_):
        arr.append([x.columns[ind], val])
        # print(str(x.columns[ind]) + ": " + str(val))
    return pd.DataFrame(arr, columns=["Column", "Relieff" + name])


def informationgain(x, y):
    arr = []
    ig = mutual_info_classif(x.values, y, discrete_features=True)
    for ind, val in enumerate(ig):
        arr.append([x.columns[ind], val])
    return pd.DataFrame(arr, columns=["Column", "Information Gain"])


def run(from_name, to_name):
    trans = loadTranslation()
    x, y = loadCsv(from_name)
    #x['LWKWKC03_0810_2'] = x['LWKWKC03_0810']
    f = fisher(x, y)
    f_f = fisher_sc(x, y)
    ig = informationgain(x, y)
    r = relieff(x, y)

    comb = f.merge(r, on="Column")
    res = comb.merge(ig, on="Column")
    res = res.merge(f_f, on="Column")
    res = pd.merge(res, trans, left_on=["Column"], right_on=["name"], how="left")
    res.drop(columns=["name"], inplace=True)

    res.to_csv(to_name)
    print("done")

def validateMultipleRuns(from_name):
    x, y = loadCsv(from_name)
    data = pd.DataFrame(x.columns, columns=["Column"])
    for i in range(1, 10):
        r = relieff(x, y, "_" + str(i))
        data = data.merge(r, on="Column")
    print(data)

def validateMultipleRunsFisher(from_name):
    x, y = loadCsv(from_name)
    data = pd.DataFrame(x.columns, columns=["Column"])
    for i in range(1, 10):
        r = fisher_sc(x, y, "_" + str(i))
        data = data.merge(r, on="Column")
    print(data)

#validateMultipleRuns("../Data/verf_neu/20220111/no_profile_nodup_desc_raw/multi/raw_data.csv")
#validateMultipleRunsFisher("../Data/verf_neu/20220111/no_profile_nodup_desc_raw/multi/raw_data.csv")
#run("../Data/verf_neu/20220120/no_profile_nodup_desc_more_raw/multi/raw_data.csv", "2022120_nodup_desc_doublelwkn_more.csv")
#run("../Data/verf_neu/20220124/nodup_hulp_desc_more_smote_fs_def_40/multi/raw_data.csv", "20220201_test_test.csv")
run("../Data/verf_neu/20220223/nodup_nohulp_nosmode/multi/raw_data.csv", "20220223_nodup_nohulp_nosmode_raw_fill.csv")
#validateMultipleRuns("../Data/verf_neu/20220120/no_profile_nodup_desc_smote_more_raw/multi/raw_data.csv")
#validateMultipleRunsFisher("../Data/verf_neu/20220120/no_profile_nodup_desc_smote_more_raw/multi/raw_data.csv")
#run("../Data/verf_neu/20220108/grouped_nohulp/multi/raw_data.csv", "20220108_grouped_nohulp.csv")
#run("../Data/verf_neu/20220108/dupe_nohulp/multi/raw_data.csv", "20220108_dupe_nohulp.csv")
