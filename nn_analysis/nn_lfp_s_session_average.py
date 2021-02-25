import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))

from tools import spike_tools as st
import h5py
from tools import nn_tool as nt
import parameters as pm
import pickle
import numpy as np
import statsmodels.stats.api as sms

sys.path.pop(1)

'''
This is for having averaged LFP signal for a single animal data with 3 s/ 1 
s 1 s/ 1 s NN/WN condition
'''

d_lfp = snakemake.input[0]
# d_lfp = "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data" \
#         "/dataset/lfp_fil/cr90_201023_lfp_fil.hdf5"
'''
setup
'''
la = {}  # lfp session average
lc = {}  # confidence interval
names = ['nn_3_1', 'nn_wn_3_1', 'nn_1_1', 'nn_wn_1_1']
db = ['70', '60']

'''
data retrieve
'''
# lfp
maxs = {}
mins = {}
with h5py.File(d_lfp, 'r') as h_5:
    maxa = {}
    mina = {}

    for n in names:
        la[n] = {}
        lc[n] = {}
        maxa[n] = {}
        mina[n] = {}
        for i in range(len(h_5[n])):
            s = db[i]

            la[n][s] = []
            lc[n][s] = []
            maxa[n][s] = []
            mina[n][s] = []

            lfp = h_5[n][i]
            lst = nt.lfp_seg(lfp, n)

            for j in range(len(lst)):
                la[n][s].append(np.mean(lst[j][:-1], axis=0))
                lst2 = np.transpose(lst[j])
                lcs = []
                for j2 in range(len(lst2)):
                    lcs.append(
                        np.array(
                            sms.DescrStatsW(
                                lst2[j2]
                                ).tconfint_mean()
                            )
                        )
                lc[n][s].append(lcs)
            print(np.shape(lc[n][s]))
            maxa[n][s].append(np.max(la[n][s]))
            mina[n][s].append(np.min(la[n][s]))

    for i in [0, 2]:
        for j in range(len(h_5[n])):
            s = db[j]
            maxs[s] = []
            mins[s] = []

            maxs[s] = np.append(maxs[s],
                                np.max([maxa[names[i]][s],
                                       maxa[names[i + 1]][s]
                                        ])
                                )
            mins[s] = np.append(mins[s],
                                np.min([mina[names[i]][s],
                                       mina[names[i + 1]][s]
                                        ])
                                )

d_lfp = {
        'la': la,
        'lc': lc,
        'maxs': maxs,
        'mins': mins
        }
#
with open(snakemake.output[0], 'wb') as datas:
    pickle.dump(d_lfp, datas, protocol=pickle.HIGHEST_PROTOCOL)

# with open("test.pk", 'wb') as datas:
#     pickle.dump(d_lfp, datas, protocol=pickle.HIGHEST_PROTOCOL)
