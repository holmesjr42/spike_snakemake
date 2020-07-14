import icsd
import numpy as np
import spike_tools as st
# import nn_tool as nt
import parameters as pm
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from matplotlib import ticker
# import seaborn as sns
# import quantities as pq
# import neo.core

# import pandas as pd
# used icsd.py to get raw csd. If you use estimated_csd or elephant,
# which gives you the output of estimated_csd, you will have filtered csd
# instead of raw one, which is different from what your supervisor asked you
# to calculate. But maybe it's not bad to study how to get estimated_csd.
# '''
# This is for a single animal data
# '''
# file_path = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#             '/data/dataset/'
#
# animal = 'cr50_190724'
#
# d_lfp = file_path + 'lfp_fil/' + animal + '_lfp_fil.hdf5'
# d_spk = file_path + 'tsp/' + animal + '_tsp.hdf5'
# sr_lfp = 1061.5 * pq.Hz
# # k = neo.core.RecordingChannel
# f = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
#     '/dataset/nn_lfp_avg/cr50_190724_nn_3_1_lfp_a.pk'
# with open('/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data'
#           '/dataset/nn_lfp_avg/cr50_190724_nn_3_1_lfp_a.pk', 'rb') as dat:
#     datas = pickle.load(dat)
#
# la = np.array(datas['la']['nn_3_1']) * pq.mV
# lfp_data = neo.core.AnalogSignal(la.T, sampling_rate=sr_lfp)
# # df = pd.DataFrame(la)
# # df.to_csv('matlab.csv')
# coord = np.arange(0, 1501, 100) * pq.micrometer
# sigma = 1 / (100 * pq.micrometer)  # I don't know how this works, but the
# # approximation of raw csd is (V[z-h] -2 * V[z]**2 + V[z+h])/h ** 2,
# # which removes the conductance out of the formula. So instead of using the
# # default sigma, I changed the value to 1/h, because in icsd.py,
# # h is divided only one time.
#
# std_input = {
#     'lfp': la,
#     'coord_electrode': coord,
#     'sigma': sigma,
#     'f_type': 'gaussian',
#     'f_order': (3, 1),
#     }
# std_input2 = {
#     'lfp': lfp_data,
#     'coord_electrode': coord,
#     'sigma': sigma,
#     'f_type': 'gaussian',
#     'f_order': (3, 1),
#     }
# std_input3 = {
#     'lfp': lfp_data,
#     'coord_electrode': coord,
#     'sigma': sigma,
#     'f_type': 'hamming',
#     'f_order': 3,
#     }
# std_csd = icsd.StandardCSD(**std_input)
# csds = std_csd.get_csd()
# csds.units = pq.mV/(pq.mm**2)
# csds = csds[1:-1, ]
#
# std2 = icsd.estimate_csd(**std_input2)
# std3 = np.array(std2) #* std2[0].units
# # std3.units = pq.mV/(pq.mm**2)
# std4 = std2[0].magnitude*std2[0].units
# std5 = std2[1].magnitude*std2[0].units
# std4.units = pq.mV/(pq.mm**2)
# std5.units = pq.mV/(pq.mm**2)
# std6 = std4.transpose()
# std7 = std5.transpose()
# # print(np.size(std4[0]), std5[0])
#
# std12 = icsd.estimate_csd(**std_input3)
# std13 = np.array(std12) #* std2[0].units
# # std3.units = pq.mV/(pq.mm**2)
# std14 = std12[0].magnitude*std12[0].units
# std15 = std12[1].magnitude*std12[0].units
# std14.units = pq.mV/(pq.mm**2)
# std15.units = pq.mV/(pq.mm**2)
# std16 = std14.transpose()
# std17 = std15.transpose()
# # use hamming filter
# plt.figure()
# plt.plot(np.arange(4353), std6[10])
# plt.plot(np.arange(4353), std7[10], color='black')
# plt.plot(np.arange(4353), std17[10], color='orange')
# plt.show()

# from spectrum import *
#
# data = data_cosine(N=2048, A=0.1, sampling=1024, freq=200)
# # If you already have the DPSS windows
# [tapers, eigen] = dpss(2048, 2.5, 4)
# # res = pmtm(data, e=tapers, v=eigen, show=False)
# # You do not need to compute the DPSS before end
# # res = pmtm(data, NW=2.5, show=False)
# res = pmtm(data, NW=2.5, k=4, show=True)
# print(res[2][0])

a= np.arange(5)
print(a)