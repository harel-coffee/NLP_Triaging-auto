import pandas as pd
import numpy as np

data = pd.read_csv("Data/output_neu_verweis.csv", sep=';', low_memory=False, decimal=',', encoding="ansi")
vraag_verwijzer = data[data['Vraag Verwijzer Category'].notnull()]
hulpvraag = data[data['Hulpvraag category'].notnull()]
vraag_verwijzer_notreatment = vraag_verwijzer[(vraag_verwijzer['Treatmentcategory'] == ' ') & (vraag_verwijzer['Treatmentcategory'] == ' ')]
hulpvraag_notreatment = hulpvraag[(hulpvraag['Treatmentcategory'] == ' ') & (hulpvraag['Extracted Category'] == ' ')]
unique_id_vv = np.unique(vraag_verwijzer_notreatment['PAID'])
unique_id_hv = np.unique(hulpvraag_notreatment['PAID'])
print("test")
