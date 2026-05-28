import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from RHom import splithalf, dir_proj, omni_sample, bypc
from RHom.visualization.rhomplots import plot_omni, plot_bypc
from RHom.visualization.pcastats import plot_grouped_scree
from RHom.preprocessing.preliminary import parallel_analysis

df = pd.read_csv('dailylife_esq.csv')

dailylife_esq = df.loc[:, "Task":"Modality"]
dailylife_esq['dataset'] = df['dataset']
dailylife_esq['id'] = df['ID']
dailylife_esq = dailylife_esq.dropna()


# # If not specifying a grouping variable, remember to specify only the data to be decomposed
# splithalf_df = dailylife_esq.iloc[:, 2:11]

# split_results = splithalf(df = splithalf_df,
#                           npc = 4,
#                           rotation = "promax",
#                           boot = 1000,
#                           file_prefix = "example_splithalf",
#                           save = False)

# # When conducting a direct-projection reproducibility analysis remember to specify the grouping variable whose levels you're comparing
# dirproj_df = df.iloc[:,2:11]
# dirproj_df['group'] = df['grouping variable']

# dirproj_results = dir_proj(df = dailylife_esq,
#                            group = "dataset",
#                            cluster = "id",
#                            npc = 3,
#                            rotation = "varimax",
#                            folds = 5,
#                            file_prefix = "testrun_uberdata")

# # An omnibus-sample reproducibility analysis can provide an alternative way of determining how robustly disparately sampled data can be blended
# omsamp_results = omni_sample(df = dailylife_esq,
#                              group = 'dataset',
#                              cluster = 'id',
#                              npc = 4,
#                              rotation = "varimax",
#                              boot = 1000,
#                              subspace = True,
#                              file_prefix = "testrun_uberdata")


# # If split-half reliability is strong enough, you can examine omnibus-sample reproducibility on a by-component level.
# bypc_results = bypc(df = dailylife_esq,
#                     group = 'dataset',
#                     cluster = 'id',
#                     npc = 3,
#                     rotation = "varimax",
#                     file_prefix = "testrun_uberdata")

plotdf = pd.read_csv('results/testrun_uberdata/testrun_uberdata_bypc_14D_3PC.csv')
plotloadings = pd.read_csv('results/testrun_uberdata/testrun_uberdata_loadings_14D_3PC.csv')

fig = plot_bypc(plotdf, plotloadings, metric="rhm")
plt.show()
fig.savefig('results/testrun_uberdata/testrun_uberdata_bypc_14D_3PC_rhm.png')
plt.close()
