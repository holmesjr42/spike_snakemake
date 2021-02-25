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

# d_spk = snakemake.input.tsp

d_spk = "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data" \
        "/dataset/tsp/cr50_190724_tsp.hdf5"

'''
setup
'''

chs = [4, 6, 8, 10, 12]
val = pm.Setup(cond2='tsp', ch=chs)
cb = val.values()[-2]
depth = val.values()[6]
dcmap = val.values()[7]
color = [dcmap[0], dcmap[1]]
names = ['nn_3_1', 'nn_wn_3_1']
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

# spike
with h5py.File(d_spk, 'r') as h_5:
    for n in names:
        mua = h_5[n][0]
        nn = nt.index_spike(mua)
        nrastor[n], npsth[n], nsmooth[n], ntime[n] = nt.psth(nn, n)

s_max = np.around(np.max([
    np.max(nsmooth[names[0]]), np.max(nsmooth[names[1]])
    ]) * 1.2, decimals=-2)

'''
plot setup
'''
# c_fig, axs = plt.subplots(len(chs), 3, figsize=(16, 9), dpi=300)
# c_fig.subplots_adjust(right=0.85, hspace=0.4, wspace=0.07)
c_loc = [0, 2]
# c_fig.patch.set_facecolor(cb)
c_fig = plt.figure(0, figsize=(16, 9), dpi=300)
# c1_fig.patch.set_facecolor(cct)

c_axs1 = c_fig.subplots(len(chs), 1, gridspec_kw=dict(left=0.1, right=0.4,
                                                      top=0.85, bottom=0.1,
                                                      ))

c_axs2 = c_fig.subplots(len(chs), 1, gridspec_kw=dict(left=0.4, right=0.5,
                                                      top=0.85, bottom=0.1,
                                                      ))
c_axs3 = c_fig.subplots(len(chs), 1, gridspec_kw=dict(left=0.5, right=0.8,
                                                      top=0.85, bottom=0.1,
                                                      ))
axs = [c_axs1, c_axs2, c_axs3]

# spk
xticks = np.array([
    np.linspace(0, 4, 5, endpoint=True),
    np.linspace(0, 4, 9, endpoint=True)
    ], dtype=object)
yticks_p = np.array([
    np.linspace(0, s_max, 3, endpoint=True),
    np.linspace(0, s_max, 9, endpoint=True)
    ], dtype=object)

for i in range(len(chs)):
    p = chs[i]
    # axs[i, 1].axis('off')
    for j in range(len(names)):

        ns = names[j]
        axp = axs[c_loc[j]][i]

        # spk
        nt.psth_plot(
            axp, ntime[ns], nsmooth[ns][p], xticks[0], yticks_p[0], ns
            )
        axp.set_xlim(-0.1, 4.1)
        axp.set_ylim([0, s_max])
        axp.tick_params(axis='y', which='minor', left=True, labelsize=4,
                        length=1.5,
                        width=0.5)
        axp.set_yticks(yticks_p[1], minor=True)
        axp.set_xticks(xticks[1], minor=True)
        axp.grid(which='both')

        for pt in np.array([0, 3.0, 4.0]):
            axp.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)

        if i + 1 == len(chs):
            pass
        else:
            axp.xaxis.set_ticklabels([])

        if j == 1:
            axp.yaxis.set_ticklabels([])
        else:
            pass
    plt.delaxes(c_axs2[i])

'''
text and axis limit
'''

c_axs1[0].set_title('Notched Noise', fontsize=15)
c_axs3[0].set_title('White Noise', fontsize=15)
# axs[0, 3].set_title('Notched Noise')
# axs[0, 4].set_title('White Noise')
c_fig.text(0.26, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
c_fig.text(0.66, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
c_fig.text(0.45, 0.95, 'Averaged PSTH', ha='center', va='center',
           fontsize=20, fontweight='bold'
           )
c_fig.text(0.05, 0.47, 'Firing rate (spk/s)', ha='center', va='center',
           rotation='vertical', fontsize=15)
# c_fig.text(0.51, 0.5, 'Amplitude (mV)', ha='center', va='center',
#            rotation='vertical', fontsize=15)

for i in range(len(chs)):
    c_axs3[i].text(1.3, 0.5, depth[i] + ' $\mu$m',
                   horizontalalignment='center',
                   verticalalignment='center',
                   # rotation='vertical',
                   transform=c_axs3[i].transAxes,
                   fontdict={'size': 13}
                   )
    if i == 0:
        c_axs3[i].text(1.3, 1.12, 'Depth',
                       horizontalalignment='center',
                       verticalalignment='center',
                       # rotation='vertical',
                       transform=c_axs3[i].transAxes,
                       fontdict={'weight': 'bold', 'size': 15}
                       )

# plt.savefig(snakemake.output[0], dpi=300)
plt.savefig('testing', dpi=300)
