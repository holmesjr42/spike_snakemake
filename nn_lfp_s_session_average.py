import spike_tools as st
import h5py
import nn_tool as nt
import parameters as pm
import pickle
import numpy as np


'''
This is for having averaged LFP signal for a single animal data with 3 s/ 1 
s NN/WN condition
'''

d_lfp = snakemake.input[0]

'''
setup
'''
ls = {}
la = {}
lr = []
names = ['nn_3_1', 'nn_wn_3_1', 'nn_1_1', 'nn_wn_1_1']

'''
data retrieve
'''
# lfp

with h5py.File(d_lfp, 'r') as h_5:
    maxa = []
    mina = []
    maxr = []
    minr = []
    for n in names:
        lfp = h_5[n][0]
        lst = nt.lfp_seg(lfp, n)
        la[n] = []
        for i in range(len(lst)):
            la[n].append(np.mean(lst[i][:-1], axis=0))
        maxa.append(np.max(la[n]))
        mina.append(np.min(la[n]))

    maxs = np.max(maxa)
    mins = np.min(mina)

d_lfp = {
        'la': la,
        'maxs': maxs,
        'mins': mins
        }

with open(snakemake.output[0], 'wb') as datas:
    pickle.dump(d_lfp, datas, protocol=pickle.HIGHEST_PROTOCOL)
