import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))

import numpy as np
import h5py
from tools import nn_tool as nt
from parameters import parameters as pm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

sys.path.pop(1)

'''
This is for a single animal data
plots psth and averaged raw lfp in the same figure
'''

d_lfp = snakemake.input.lfp
d_spk = snakemake.input.tsp

# d_lfp = "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data" \
#         "/dataset/nn_lfp_avg/cr50_190724_nn_lfp_a.pk"
# d_spk = "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data" \
#         "/dataset/tsp/cr50_190724_tsp.hdf5"
#
'''
setup
'''

chs = [4, 6, 8, 10, 12]
val = pm.Setup(cond2='tsp', ch=chs)
cb = val.values()[-2]
depth = val.values()[6]
dcmap = val.values()[7]
color = [dcmap[0], dcmap[1]]
names = ['nn_1_1', 'nn_wn_1_1']
sns.set_style('ticks')

# spk
f = {}
sm = {}
sl = {}
nrastor = {}
npsth = {}
nsmooth = {}
ntime = {}

'''
data retrieve
'''
# lfp
with open(d_lfp, 'rb') as datas:
    dat = pickle.load(datas)
    la = dat['la']
    maxs = dat['maxs']
    mins = dat['mins']

# spike
with h5py.File(d_spk, 'r') as h_5:
    for n in names:
        mua = h_5[n][0]
        nn = nt.index_spike(mua)
        nrastor[n], npsth[n], nsmooth[n], ntime[n] = nt.psth(nn, n)

s_max = np.around(np.max([
    np.max(nsmooth[names[0]]), np.max(nsmooth[names[1]])
    ]), decimals=-2) + 100



'''
plot setup
'''
c_fig, axs = plt.subplots(len(chs), 5, figsize=(16, 9), dpi=300)
c_fig.subplots_adjust(right=0.85, hspace=0.4, wspace=0.07)
# c_fig.patch.set_facecolor(cb)

# lfp
# min and max value for x, y limit
maxf = np.around(maxs, decimals=-2) + 100
minf = np.around(mins, decimals=-2) - 100

# spk
xticks = np.array([
                   np.linspace(0, 2, 5, endpoint=True),
                   np.linspace(0, 2, 9, endpoint=True)
                  ], dtype=object)

yticks_p = np.array([
                     np.linspace(0, s_max, 3, endpoint=True),
                     np.linspace(0, s_max, 9, endpoint=True)
                    ], dtype=object)

yticks_l = np.array([
                     np.linspace(minf, maxf, 3, endpoint=True),
                     np.linspace(minf, maxf, 9, endpoint=True)
                    ], dtype=object)


for i in range(len(chs)):
    p = chs[i]
    # axs[i, 1].axis('off')
    for j in range(len(names)):

        ns = names[j]
        axl = axs[i, j+3]
        laf = la[ns][p]
        axp = axs[i, j]

        # lfp
        lt = np.linspace(0, 2100, num=len(laf), endpoint=True)
        nt.lfp_plot(axl, lt, laf, xticks[0], yticks_l[0], ns)
        axl.set_xlim(-0.1, 2.1)
        axl.set_ylim([minf, maxf])
        axl.set_yticks(yticks_l[1], minor=True)
        axl.set_xticks(xticks[1], minor=True)
        axl.tick_params(axis='y', which='minor', left=True, labelsize=4,
                        length=1.5,
                        width=0.3)
        axl.grid(which='both')
        # axl.yaxis.tick_right()
        axl.yaxis.set_label_position("right")

        # spk
        nt.psth_plot(
                    axp, ntime[ns], nsmooth[ns][p], xticks[0], yticks_p[0], ns
                    )
        axp.set_xlim(-0.1, 2.1)
        axp.set_ylim([0, s_max])
        axp.tick_params(axis='y', which='minor', left=True, labelsize=4,
                        length=1.5,
                        width=0.3)
        axp.set_yticks(yticks_p[1], minor=True)
        axp.set_xticks(xticks[1], minor=True)
        axp.grid(which='both')

        for pt in np.array([0, 1.0, 2.0]):
            axl.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)
            axp.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)

        if i + 1 == len(chs):
            pass
        else:
            axl.xaxis.set_ticklabels([])
            axp.xaxis.set_ticklabels([])

        if j == 1:
            axl.yaxis.set_ticklabels([])
            axp.yaxis.set_ticklabels([])
        else:
            pass

    c_fig.delaxes(axs[i, 2])

'''
text and axis limit
'''

axs[0, 0].set_title('Notched Noise')
axs[0, 1].set_title('White Noise')
axs[0, 3].set_title('Notched Noise')
axs[0, 4].set_title('White Noise')
c_fig.text(0.27, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
c_fig.text(0.71, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
c_fig.text(0.27, 0.95, 'Averaged PSTH', ha='center', va='center', fontsize=15)
c_fig.text(0.71, 0.95, 'Averaged LFP', ha='center', va='center', fontsize=15)
c_fig.text(0.08, 0.5, 'Firing rate (spk/s)', ha='center', va='center',
           rotation='vertical', fontsize=15)
# c_fig.text(0.47, 0.5, 'Amplitude (mV)', ha='center', va='center',
#            rotation='vertical', fontsize=15)
c_fig.text(0.51, 0.5, 'Amplitude (mV)', ha='center', va='center',
           rotation='vertical', fontsize=15)
# c_fig.text(0.91, 0.5, 'Firing rate (spk/s)', ha='center', va='center',
#            rotation='vertical', fontsize=15)

for i in range(len(chs)):
    axs[i, 4].text(3, -400, depth[i] + ' $\mu$m',
                   ha='center', va='center', fontsize=10)
    if i == 0:
        axs[i, 4].text(3, 1000, 'Depth', ha='center', va='center', fontsize=12)

plt.savefig(snakemake.output[0], dpi=300)
# plt.savefig('testing', dpi=300)