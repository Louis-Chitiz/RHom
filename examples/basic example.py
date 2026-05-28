import pandas as pd
from RHom import basePCA

# konu21 = pd.read_csv("konu2021.csv")

# konu_esq = konu21.loc[:, "Focus":"Source"]
# konu_esq = konu_esq.rename(columns={'Focus':"Task",
#                            "Other":"People"})

# konu_esq['dataset'] = 'Konu'
# konu_esq['ID'] = konu21['Participant_number']
# konu_esq = konu_esq.dropna()

# # Park the participant ID on the index so it isn't treated as a numeric feature
# konu_esq = konu_esq.set_index('ID')

df = pd.read_csv('dailylife_esq.csv')

dailylife_esq = df.loc[:, "Task":"Modality"]
dailylife_esq['dataset'] = df['dataset']
dailylife_esq['id'] = df['ID']
dailylife_esq = dailylife_esq.dropna()

dailylife_esq = dailylife_esq.set_index('id')


model = basePCA(n_components=3, rotation="varimax", verbosity = 1)
model.fit(dailylife_esq)

# Scores are now on model.extra_columns alongside any metadata columns (e.g. 'dataset')
model.save(path="results", pathprefix="uberdata_3PC")





