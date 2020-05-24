import glob
import os.path as opath
import scipy.io as spio
import numpy as np
import matplotlib as plt
import h5py
import spike_tools as st

'''
variable
'''

list_freq = np.load('/home/bhc/codes/spikes_classics/parameters/list_freq'
                    '.npy', allow_pickle=True)
p_info = np.load('/home/bhc/codes/spikes_classics/parameters'
                 '/time_index_multi_tones_500ms_180s.npy', allow_pickle=True)
with h5py.File('/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021'
               '/data/cr50_190724/cr50_190724_tsp.hdf5',
               'r') as h_5:
    time_snip = h_5['nnm_3_0'][0]
    mua = st.index_strf(time_snip)

    k = st.tsp_strf(p_info, mua)
    print(k[1][10][7])