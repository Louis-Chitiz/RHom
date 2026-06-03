import pandas as pd
from RHom import rhom
from RHom import basePCA

data = pd.read_csv("examples/output.csv")

model = basePCA(n_components=4,rotation="varimax")

projected_results = model.rhom.cv(data)
model.save(path="results",pathprefix="PCA_results")


