import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from RHom import splithalf, dir_proj, omni_sample, omsamp_bypc, dir_proj_bypc, holdout_cv, splithalf_bypc, consensus_pca

df = pd.read_csv('dailylife_esq.csv')

dailylife_esq = df.loc[:, "Task":"Deliberate"]
feat_cols = dailylife_esq.columns
dailylife_esq['dataset'] = df['dataset']
dailylife_esq['id'] = df['ID']
dailylife_esq = dailylife_esq.dropna()

# dailylife_esq = dailylife_esq[dailylife_esq.dataset == "Poerio1"]

# # Bootstrap mode — cluster-aware so participants are resampled as units
# result = consensus_pca(
#     df=dailylife_esq, boot=1000, cluster='id', stratify = 'dataset',
#     npc=4, shuffle = False, method='eigen', rotation='varimax', corr='spearman',
#     file_prefix='uberdata_consensus_boot1000',
# )

# result = consensus_pca(df=dailylife_esq, group='dataset', cluster='id', npc = 4,
#                        method = 'eigen', corr = 'spearman', rotation = 'varimax',
#                        file_prefix = "uberdata_consensus_LOGO")

# # CV mode — leave-one-source-out, gives source-balanced consensus
# result = consensus_pca(
#     df=dailylife_esq, group='dataset', cluster='id', npc=4,
#     method='eigen', rotation='varimax', corr='spearman',
#     file_prefix='consensus_logo',
# )

# Stratified k-fold (LOGO-style guarantee that each fold contains all sources)
# result = consensus_pca(
#     df=dailylife_esq, group='dataset', folds=5, cluster='id', npc=4,
#     method='eigen', rotation='varimax', corr='spearman',
#     file_prefix='consensus_strat5',
# )


# from RHom.preprocessing.preliminary import parallel_analysis
# npc_pooled = parallel_analysis(dailylife_esq[feat_cols])
# npc_per_source = {g: parallel_analysis(dailylife_esq.loc[dailylife_esq.dataset == g, feat_cols], title = g)
#                   for g in dailylife_esq.dataset.unique()}

# for source in npc_per_source.keys():
#     npc_per_source[source]['figure'].savefig(f'results/pa_per_group/parallel_analysis_{source}.png')

# plt.close('all')


# Leave-one-group-out
# holdout_cv(df=dailylife_esq, group='dataset', cluster='id', npc=2,
#            method='eigen', corr='spearman', subspace=True, shuffle=True,
#            file_prefix='logo_shuffle_test')

# ho_df = pd.read_csv('results/logo_test/logo_test_holdout_cv_14D_2PC.csv')

# from RHom.visualization.rhomplots import plot_omni
# fig = plot_omni(ho_df, metric="sub", chance=0.65)
# plt.show()
# fig.savefig('results/logo_test/logo_test_holdout_cv_14D_2PC_sub.png')
# plt.close()

# from RHom import omni_variance
# df_var = omni_variance(df=dailylife_esq, group='dataset', cluster='id',
#                        npc=3, method='eigen', rotation='varimax', corr='spearman',
#                        file_prefix='testrun_uberdata')

# dailylife_esq = dailylife_esq.loc[df['dataset'].isin(['Nerissa', 'Poerio1', 'Poerio2', 'Poerio3'])]

# # If not specifying a grouping variable, remember to specify only the data to be decomposed
# splithalf_df = dailylife_esq.iloc[:, 2:11]

# split_results = splithalf(df = dailylife_esq,
#                           npc = 4,
#                           cluster = "id",
#                           stratify = "dataset",
#                           method = "eigen",
#                           corr = "spearman",
#                           rotation = "varimax",
#                           shuffle = True,
#                           boot = 1000,
#                           file_prefix = "uberdata_splithalf_shuffle")

# split_bypc = splithalf_bypc(df = dailylife_esq,
#                             npc = 4,
#                             method = "eigen",
#                             corr = "spearman",
#                             rotation = "varimax",
#                             cluster = "id",
#                             stratify = "dataset",
#                             shuffle = True,
#                             file_prefix = "uberdata_splithalf_shuffle")

# # When conducting a direct-projection reproducibility analysis remember to specify the grouping variable whose levels you're comparing
# dirproj_df = df.iloc[:,2:11]
# dirproj_df['group'] = df['grouping variable']

# dirproj_results = dir_proj(df = dailylife_esq,
#                            group = "dataset",
#                            cluster = "id",
#                            npc = 3,
#                            rotation = "varimax",
#                            method = "eigen",
#                            corr = "spearman",
#                            subspace = True,
#                            folds = 5,
#                            file_prefix = "testrun_uberdata")

# ## By-Component direct projection can be used to examine the reproducibility of individual components across groups,
# ## which can be especially useful when the number of components is large and/or when some components are expected to be more robust than others.
# dirprojbypc = dir_proj_bypc(df=dailylife_esq, group='dataset', cluster='id',
#                             npc=3, method="eigen", corr='spearman', rotation='varimax',
#                             file_prefix='testrun_uberdata')

# # # An omnibus-sample reproducibility analysis can provide an alternative way of determining how robustly disparately sampled data can be blended
# omsamp_results = omni_sample(df = dailylife_esq,
#                              group = 'dataset',
#                              cluster = 'id',
#                              npc = 3,
#                              method = "eigen",
#                              corr = "spearman",
#                              rotation = "varimax",
#                              boot = 1000,
#                              subspace = True,
#                              file_prefix = "testrun_uberdata")


# # # If split-half reliability is strong enough, you can examine omnibus-sample reproducibility on a by-component level.
# bypc_results = omsamp_bypc(df = dailylife_esq,
#                     group = 'dataset',
#                     cluster = 'id',
#                     npc = 3,
#                     method = "eigen",
#                     corr = "spearman",
#                     rotation = "varimax",
#                     file_prefix = "testrun_uberdata")


# plotdf = pd.read_csv('results/testrun_uberdata/testrun_uberdata_bypc_14D_3PC.csv')
# plotloadings = pd.read_csv('results/testrun_uberdata/testrun_uberdata_loadings_14D_3PC.csv')

# fig = plot_bypc(plotdf, plotloadings, metric="rhm")
# plt.show()
# fig.savefig('results/testrun_uberdata/testrun_uberdata_bypc_14D_3PC_rhm.png')
# plt.close()