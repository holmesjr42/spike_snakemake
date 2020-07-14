from matplotlib import pyplot
# import icsd
# import numpy as np
# import parameters as pm
# import pickle
# from matplotlib import colors

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import seaborn as sns
# import quantities as pq
# import neo.core
# import sys

# '''
# This is for a single animal data
# '''
# sns.set_style("ticks")
# # file_input = snakemake.input[0]
# file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#              '/data/dataset/nnm_lfp_avg/cr29_190228_lfp_avg_nnm.pk'
#
# sr_lfp = 1061.5 * pq.Hz  # for convert time into data coord in the data file
#
# with open(file_input, 'rb') as dat:
#     datas = pickle.load(dat)
#
# a = datas['nnm_3_0']['70']
# print(np.shape(a))