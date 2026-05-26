import pandas as pd
from RHom.pca import basePCA

konu21 = pd.read_csv("konu2021.csv")

konu_esq = konu21.loc[:, "Focus":"Source"]
konu_esq = konu_esq.rename(columns={'Focus':"Task",
                           "Other":"People"})

konu_esq['dataset'] = 'Konu'
konu_esq['ID'] = konu21['Participant_number']
konu_esq = konu_esq.dropna()

model = basePCA(n_components=4,rotation="varimax")



projected_results = model.fit_transform(konu_esq)
model.save(path="results",pathprefix="PCA_results")




