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
import quantities as pq
import pylustrator

# pylustrator.start()
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
animal = d_spk.split('/')[-1].split('_')[0]
dbs = ['70', '60']
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
temp = ['testing0', 'testing1']
'''
data retrieve
'''
if animal == 'cr31' or animal == 'cr33':
    plt.figure()
    plt.savefig(snakemake.output[1], dpi=150)
    db = [0]
else:
    db = [0, 1]

for d in db:
    s = dbs[d]
    # lfp
    with open(d_lfp, 'rb') as datas:
        dat = pickle.load(datas)
        la = dat['la']
        lc = dat['lc']
        maxs = dat['maxs']
        mins = dat['mins']

    # maximum/minimum for ticks
    ti = nt.lfp_index(2800)
    tf = nt.lfp_index(4000)
    lnn_m = np.transpose(la[names[0]][s])[ti:tf] * pq.microvolt
    lwn_m = np.transpose(la[names[1]][s])[ti:tf] * pq.microvolt

    lnn_m.units = pq.mV
    lwn_m.units = pq.mV

    maxf3 = np.around(
        np.max([np.max(lnn_m),  np.max(lwn_m)]), decimals=1
        )
    minf3 = np.around(
        np.min([np.min(lnn_m),  np.min(lwn_m)]), decimals=1
        )

    yticks_l1 = np.array([
        np.array([minf3, 0, maxf3]),
        np.array([minf3, np.around(minf3/2, decimals=1), 0, maxf3])
        ], dtype=object)

    # spike
    with h5py.File(d_spk, 'r') as h_5:
        for n in names:
            mua = h_5[n][d]
            nn = nt.index_spike(mua)
            nrastor[n], npsth[n], nsmooth[n], ntime[n] = nt.psth(nn, n)

            nsmooth[n] = nsmooth[n] * pq.ms**(-1)
            nsmooth[n].units = pq.s ** (-1)

    s_max4 = np.around(np.max([
        np.max(nsmooth[names[0]]), np.max(nsmooth[names[1]])
        ]) * 1.3, decimals=-2)

    s_max3 = np.around(np.max([
        np.max(np.transpose(nsmooth[names[0]])[2800:3900]),
        np.max(np.transpose(nsmooth[names[1]])[2800:3900])
        ]) * 1.3, decimals=-1)

    '''
    plot setup
    '''
    c_fig, axs = plt.subplots(len(chs), 5, figsize=(16, 9))
    c_fig.subplots_adjust(right=0.85, hspace=0.4, wspace=0.35)
    # c_fig.patch.set_facecolor(cb)

    # lfp
    # min and max value for x, y limit
    maxf4 = maxs[s][0] * pq.microvolt
    minf4 = mins[s][0] * pq.microvolt
    maxf4.units = pq.mV
    minf4.units = pq.mV
    maxf4 = np.around(maxf4, decimals=1)
    minf4 = np.around(minf4, decimals=1)

    # spk
    xticks2 = np.array([
        np.linspace(0, 4, 5, endpoint=True),
        np.linspace(0, 4, 9, endpoint=True)
        ], dtype=object)

    xticks1 = np.array([
        np.linspace(3, 4.0, 3, endpoint=True),
        np.linspace(3, 4.0, 9, endpoint=True)
        ], dtype=object)

    yticks_p2 = np.array([
        np.linspace(0, s_max4, 3, endpoint=True),
        np.linspace(0, s_max4, 9, endpoint=True)
        ], dtype=object)

    yticks_p1 = np.array([
        np.linspace(0, s_max3, 3, endpoint=True),
        np.linspace(0, s_max3, 9, endpoint=True)
        ], dtype=object)

    yticks_l2 = np.array([
        [minf4, 0, maxf4],
        [minf4, np.around(minf4/2, decimals=1), 0, maxf4]
        ], dtype=object)

    for i in range(len(chs)):
        p = chs[i]
        # axs[i, 1].axis('off')

        nn = names[0]
        wn = names[1]
        axp4 = axs[i, 0]
        axp3 = axs[i, 1]
        axl4 = axs[i, 3]
        axl3 = axs[i, 4]
        lafn = la[nn][s][p] * pq.microvolt
        lafw = la[wn][s][p] * pq.microvolt
        lacn = lc[nn][s][p] * pq.microvolt
        lacw = lc[wn][s][p] * pq.microvolt

        lafn.units = pq.mV
        lafw.units = pq.mV
        lacn.units = pq.mV
        lacw.units = pq.mV

        axl4.axhline(y=0, color='#737373', linestyle='-', lw=1, alpha=0.8)
        axl3.axhline(y=0, color='#737373', linestyle='-', lw=1, alpha=0.8)

        # lfp whole
        lt = np.linspace(0, 4100, num=len(lafn), endpoint=True)
        nt.lfp_plot(axl4, lt, lafn, lacn, xticks2[0], yticks_l2[0], nn)
        nt.lfp_plot(axl4, lt, lafw, lacw, xticks2[0], yticks_l2[0], wn)
        axl4.set_xlim(-0.1, 4.1)
        # axl4.set_ylim([minf4, maxf4])
        axl4.set_yticks(yticks_l2[1], minor=True)
        axl4.set_xticks(xticks2[1], minor=True)
        axl4.tick_params(axis='y', which='minor', left=True, labelsize=4,
                         length=1.5,
                         width=0.3)
        axl4.grid(which='both')
        # axl.yaxis.tick_right()
        axl4.yaxis.set_label_position("right")

        # lfp part

        lt = np.linspace(2800, 4001, num=len(lafn[ti:tf + 1]), endpoint=True)
        nt.lfp_plot(axl3, lt, lafn[ti:tf + 1], lacn[ti:tf + 1], xticks1[
            0], yticks_l1[0], nn)
        nt.lfp_plot(axl3, lt, lafw[ti:tf + 1], lacw[ti:tf + 1], xticks1[0],
                    yticks_l1[0], wn)
        axl3.set_xlim(2.8, 4.0)
        axl3.set_ylim([minf3 * 1.4, maxf3 * 1.4])
        axl3.set_yticks(yticks_l1[1], minor=True)
        axl3.set_xticks(xticks1[1], minor=True)
        axl3.tick_params(axis='y', which='minor', left=True, labelsize=4,
                         length=1.5,
                         width=0.3)
        axl4.grid(which='both')
        # axl.yaxis.tick_right()
        axl4.yaxis.set_label_position("right")

        # spk whole
        nt.psth_plot(
            axp4, ntime[nn], nsmooth[nn][p], xticks2[0], yticks_p2[0], nn
            )
        nt.psth_plot(
            axp4, ntime[wn], nsmooth[wn][p], xticks2[0], yticks_p2[0], wn
            )
        axp4.set_xlim(-0.1, 4.1)
        axp4.set_ylim([0, s_max4])
        axp4.tick_params(axis='y', which='minor', left=True, labelsize=4,
                         length=1.5,
                         width=0.3)
        axp4.set_yticks(yticks_p2[1], minor=True)
        axp4.set_xticks(xticks2[1], minor=True)
        axp4.grid(which='both')

        # spk part
        nt.psth_plot(
            axp3, ntime[nn][2800:4001], nsmooth[nn][p][2800:4001],
            xticks1[0], yticks_p1[0], nn
            )
        nt.psth_plot(
            axp3, ntime[wn][2800:4001], nsmooth[wn][p][2800:4001],
            xticks1[0], yticks_p1[0], wn
            )

        axp3.set_xlim(2.8, 4.0)
        axp3.set_ylim([0, s_max3])
        axp3.tick_params(axis='y', which='minor', left=True, labelsize=4,
                         length=1.5,
                         width=0.3)
        axp3.set_yticks(yticks_p1[1], minor=True)
        axp3.set_xticks(xticks1[1], minor=True)
        axp3.grid(which='both')

        # borderline for on/off
        for pt in np.array([0, 3.0, 4.0]):
            axl4.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)
            axp4.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)
        for pt in np.array([3.0, 4.0]):
            axl3.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)
            axp3.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.5)

        if i + 1 == len(chs):
            pass
        else:
            axl4.tick_params(labelbottom=False)
            axp4.tick_params(labelbottom=False)
            axl3.tick_params(labelbottom=False)
            axp3.tick_params(labelbottom=False)

        c_fig.delaxes(axs[i, 2])

    '''
    text and axis limit
    '''

    axs[0, 0].set_title('Noise On-Off')
    axs[0, 1].set_title('Noise Off')
    axs[0, 3].set_title('Noise On-Off')
    axs[0, 4].set_title('Noise Off')
    c_fig.text(0.27, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
    c_fig.text(0.71, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
    c_fig.text(0.27, 0.95, 'Averaged PSTH', ha='center', va='center',
               fontsize=15)
    c_fig.text(0.71, 0.95, 'Averaged LFP', ha='center', va='center',
               fontsize=15)
    c_fig.text(0.08, 0.5, 'Firing rate (spk/s)', ha='center', va='center',
               rotation='vertical', fontsize=15)
    # c_fig.text(0.47, 0.5, 'Amplitude (mV)', ha='center', va='center',
    #            rotation='vertical', fontsize=15)
    c_fig.text(0.51, 0.5, 'Amplitude (mV)', ha='center', va='center',
               rotation='vertical', fontsize=15)
    # c_fig.text(0.91, 0.5, 'Firing rate (spk/s)', ha='center', va='center',
    #            rotation='vertical', fontsize=15)

    for i in range(len(chs)):
        axs[i, 4].text(1.5, 0.5, depth[i] + ' $\mu$m',
                       horizontalalignment='center',
                       verticalalignment='center',
                       # rotation='vertical',
                       transform=axs[i, 4].transAxes,
                       fontdict={'weight': 'bold', 'size': 11}
                       )

        if i == 0:
            axs[i, 4].text(1.5, 1.1, 'Depth',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axs[i, 4].transAxes,
                           fontdict={'weight': 'bold', 'size': 11}
                           )

    plt.savefig(snakemake.output[d], dpi=300)
    # plt.savefig(temp[d], dpi=300)
    # plt.show()
    c_fig.clf()
