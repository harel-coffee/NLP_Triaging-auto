from pathlib import Path

import numpy
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

import dataUtilities

def useTopNRelieff(treatment, type, n, extra):
    features = pd.read_csv("param_analyse/" + type + ".csv")
    features = np.append(features.sort_values(by='Relieff', ascending=False)['Column'].head(n).values, extra)
    treatment = treatment[features]
    return treatment


def loadReplacedSetCrossval(target="", inputFile="", withFeatureSelection=False, featureSize=50,
                            featureFile="withagehulpverwijzer_multi"):
    Path("./Data/" + target).mkdir(parents=True, exist_ok=True)

    all = [
        'BHSTATUS1', 'BHSTOPREDEN1',  # 'BHPROFIEL1',
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
        'LWKNIH3110', 'LWKNIH3210', 'LWKNIH3310', 'LWKNIH3410', 'LWKNIH3510', 'LWKNIH3610', 'LWKNIH3710',
        'LWKNIH37_110',
        'LWKNIH3810', 'LWKNIH39_0110', 'LWKNIH39_0210', 'LWKNIH39_0310', 'LWKNIH39_0410', 'LWKNIH39_0510',
        'LWKNIH39_0610',
        'LWKNIH39_0710', 'LWKNIH39_0810', 'LWKNIH39_0910', 'LWKNIH39_1010', 'LWKNIH39_1110', 'LWKNIH39_1210',
        'LWKNIH39_11_110', 'LWKNIH4010', 'LWKNIH4110']
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
    extra = ['Age']
    misc = ['GESLACHT', 'hulpvraag_1', 'hulpvraag_2', 'hulpvraag_3', 'hulpvraag_4', 'vverwijzer_1', 'vverwijzer_2',
            'vverwijzer_3', 'vverwijzer_4', 'vverwijzer_5']
    customScores = ['Stress_arbeit_privatleben', 'alcohol_drugs', 'Depression_Index', 'Schlafqualität']
    flags = red + yellow + blue + black + misc + undetermined + all + extra + misc  # + customScores
    flags = list(dict.fromkeys(flags))  # for all and removing duplicates
    umcgPath = inputFile
    # umcgPath = './Data/umcg_treat_withhulpandverwijzer.csv'  # new with hulp and xx
    # umcgPath = './Data/umcg_treat_newdata.csv'  # new
    # umcgPath = './Data/umcg_treat.csv'  # old
    # umcgPath = './Data/umcg_treat_newdata_verweis_final.csv'  # verweise
    #    umcgPath = './Data/umcg_treat_newdata_with_age.csv'  # verweise
    cleaned = dataUtilities.cleanData(umcgPath)  # pd.read_csv(umcgPath, sep=';', low_memory=False, decimal=',')
    cleaned = cleaned.replace(r'^\s*$', np.nan, regex=True)

    doi = cleaned[flags + ['Behandlungskategorie']]

    doi['Behandlungskategorie'] = pd.to_numeric(doi['Behandlungskategorie'])
    print('DOI shape: ' + str(doi.shape))
    treatment = doi[doi['Behandlungskategorie'].notna()]
    print('Treatment shape: ' + str(treatment.shape))
    print('NaN count: ' + str(treatment.isnull().sum()))
    treatment = dataUtilities.fillOmittedValues(treatment)
    print('Filled shape: ' + str(treatment.shape))
    nullseries = treatment.isnull().sum()
    for index, item in nullseries.iteritems():
        print('Item ' + str(index) + ' has ' + str(item) + ' missing values')
    print('NaN count: ' + str(treatment.isnull().sum()))
    # treatment = dataUtilities.dimensionalityReduction(treatment) // custom scores, removes old
    print('Reduced shape: ' + str(treatment.shape))
    treatment.dropna(axis=1, inplace=True)  # drop all cols with nan
    treatment.dropna(axis=0, inplace=True)
    for colname, colvalues in treatment.iteritems():
        try:
            pd.to_numeric(colvalues)
        except:
            treatment.drop(columns=[colname], inplace=True)
    print('Dropped shape: ' + str(treatment.shape))
    # treatment.drop(columns=['Behandlungskategorie'], inplace=True)
    unique, counts = np.unique(treatment.values, return_counts=True, axis=1)
    print('Different patients: ' + str(len(unique)))
    listOfCols = list(treatment.columns)
    # with open('C:/Users/michn/PycharmProjects/Triaging/items.txt', 'w') as file:
    #    for col in listOfCols:
    #        file.write(col + '\n')
    # treatment = treatment[treatment['Behandlungskategorie'] < 6] # filter 6

    if withFeatureSelection:
        treatment = useTopNRelieff(treatment, featureFile, featureSize, ['Behandlungskategorie'])

    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] == 1] = 1
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] == 2] = 2
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] == 3] = 3
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] == 4] = np.NAN
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] == 5] = np.NAN
    treatment['Behandlungskategorie'].loc[treatment['Behandlungskategorie'] == 6] = 0
    treatment = treatment.dropna()

    print(np.unique(treatment['Behandlungskategorie'], return_counts=True))

    sm = SMOTE(random_state=42)
    y_o = treatment['Behandlungskategorie']
    x_o = treatment.drop(columns=['Behandlungskategorie'])
    x, y = sm.fit_resample(x_o, y_o)
    balanced = pd.DataFrame(x)
    balanced['Behandlungskategorie'] = y

    uni = np.unique(y)
    for i in uni:
        print('Label: ' + str(y[i]) + ' Number of Samples: ' + str(len(balanced[balanced['Behandlungskategorie'] == i])))

    balanced.to_csv('./Data/' + target + "raw_data.csv", sep=";", decimal=",")



    cross, val = dataUtilities.splitFrameCrossvalCustom(balanced)
    for index, fold in enumerate(cross):
        np.save('./Data/' + target + 'fold_' + str(index) + '_flags_data',
                fold[0].drop(columns=['Behandlungskategorie']).values)
        np.save('./Data/' + target + 'fold_' + str(index) + '_flags_label',
                fold[0]['Behandlungskategorie'].values)
        np.save('./Data/' + target + 'fold_' + str(index) + '_val_data',
                fold[1].drop(columns=['Behandlungskategorie']).values)
        np.save('./Data/' + target + 'fold_' + str(index) + '_val_label',
                fold[1]['Behandlungskategorie'].values)
    np.save('./Data/' + target + 'validation_all_data', val.drop(columns=['Behandlungskategorie']).values)
    np.save('./Data/' + target + 'validation_all_label', val['Behandlungskategorie'].values)
    print("DONE.")
    return treatment


topN = 70

#loadReplacedSetCrossval("verf_neu/20220120/no_profile_nodup_desc_smote_more_raw/multi/",
#                        "./Data/20220113_umcg_treat_hulp_noduped_desc_moreentries.csv", False,
#                        40, "20220113_nodup_desc_smote_more")

loadReplacedSetCrossval("verf_neu/20220124/nodup_hulp_desc_more_smote_fs_smote_40/multi/",
                        "./Data/20220113_umcg_treat_hulp_noduped_desc_moreentries.csv", True,
                        40, "20220113_nodup_desc_smote_more")


print("EOF")
