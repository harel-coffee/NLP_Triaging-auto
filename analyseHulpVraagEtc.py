import pandas as pd

file = "./Data/verf_neu/20220111/no_profile_nodup_asc_raw/multi/raw_data.csv"

data = pd.read_csv(file, sep=";", decimal=",")

hulp_ret = []
for behcat in range(0, 4):
    row = []
    for vv in range(1, 6):
        row.append(data[(data['vverwijzer_' + str(vv)] == 1) & (data['Behandlungskategorie'] == behcat)][
                       'Behandlungskategorie'].count())
    hulp_ret.append(row)

print(hulp_ret)


hulp_ret = []
for behcat in range(0, 4):
    row = []
    for vv in range(1, 5):
        row.append(data[(data['hulpvraag_' + str(vv)] == 1) & (data['Behandlungskategorie'] == behcat)][
                       'Behandlungskategorie'].count())
    hulp_ret.append(row)

print(hulp_ret)

