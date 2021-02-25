
######## Snakemake header ########
import sys; sys.path.extend(['/home/bhc/anaconda3/lib/python3.7/site-packages', '/home/bhc/codes/spike_snakemake']); import pickle; snakemake = pickle.loads(b'\x80\x03csnakemake.script\nSnakemake\nq\x00)\x81q\x01}q\x02(X\x05\x00\x00\x00inputq\x03csnakemake.io\nInputFiles\nq\x04)\x81q\x05(Xk\x00\x00\x00/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/dataset/nn_lfp_avg/cr35_190403_nn_lfp_a.pkq\x06Xa\x00\x00\x00/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/dataset/tsp/cr35_190403_tsp.hdf5q\x07e}q\x08(X\x06\x00\x00\x00_namesq\t}q\n(X\x03\x00\x00\x00lfpq\x0bK\x00N\x86q\x0cX\x03\x00\x00\x00tspq\rK\x01N\x86q\x0euh\x0bh\x06h\rh\x07ubX\x06\x00\x00\x00outputq\x0fcsnakemake.io\nOutputFiles\nq\x10)\x81q\x11Xs\x00\x00\x00/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/dataset/image/nn_3_1/cr35_190403_nn31_psth_lfp.pngq\x12a}q\x13h\t}q\x14sbX\x06\x00\x00\x00paramsq\x15csnakemake.io\nParams\nq\x16)\x81q\x17}q\x18h\t}q\x19sbX\t\x00\x00\x00wildcardsq\x1acsnakemake.io\nWildcards\nq\x1b)\x81q\x1cX\x0b\x00\x00\x00cr35_190403q\x1da}q\x1e(h\t}q\x1fX\x04\x00\x00\x00nameq K\x00N\x86q!sX\x04\x00\x00\x00nameq"h\x1dubX\x07\x00\x00\x00threadsq#K\x01X\t\x00\x00\x00resourcesq$csnakemake.io\nResources\nq%)\x81q&(K\x01K\x01e}q\'(h\t}q((X\x06\x00\x00\x00_coresq)K\x00N\x86q*X\x06\x00\x00\x00_nodesq+K\x01N\x86q,uh)K\x01h+K\x01ubX\x03\x00\x00\x00logq-csnakemake.io\nLog\nq.)\x81q/}q0h\t}q1sbX\x06\x00\x00\x00configq2}q3X\x04\x00\x00\x00ruleq4X\x0f\x00\x00\x00nn_3_1_lfp_plotq5X\x0f\x00\x00\x00bench_iterationq6NX\t\x00\x00\x00scriptdirq7X\x1f\x00\x00\x00/home/bhc/codes/spike_snakemakeq8ub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/nn_analysis/nn31_psth_lfp_plot.py';
######## Original script #########
import numpy as np
import h5py
from tools import nn_tool as nt
from parameters import parameters as pm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

'''
This is for a single animal data
'''

d_lfp = snakemake.input.lfp
d_spk = snakemake.input.tsp

# d_lfp = "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data" \
#         "/dataset/nn_lfp_avg/cr50_190724_nn_lfp_a.pk"
# d_spk = "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data" \
#         "/dataset/tsp/cr50_190724_tsp.hdf5"

'''
setup
'''

chs = [1, 3, 5, 7, 9, 11, 13, 14]
val = pm.Setup(cond2='tsp', ch=chs)
cb = val.values()[-1]
depth = val.values()[6]
color = ['dodgerblue', 'tab:orange']
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
c_fig.patch.set_facecolor(cb)

# lfp
# min and max value for x, y limit
maxf = np.around(maxs, decimals=-2) + 100
minf = np.around(mins, decimals=-2) - 100

# spk
xticks = np.array([
                   np.linspace(0, 4, 5, endpoint=True),
                   np.linspace(0, 4, 9, endpoint=True)
                  ])
yticks_p = np.array([
                     np.linspace(0, s_max, 3, endpoint=True),
                     np.linspace(0, s_max, 9, endpoint=True)
                    ])

yticks_l = np.array([
                     np.linspace(minf, maxf, 3, endpoint=True),
                     np.linspace(minf, maxf, 9, endpoint=True)
                    ])


for i in range(len(chs)):
    p = chs[i]
    # axs[i, 1].axis('off')
    for j in range(len(names)):

        ns = names[j]
        axl = axs[i, j+3]
        laf = la[ns][p]
        axp = axs[i, j]

        # lfp
        lt = np.linspace(0, 4100, num=len(laf), endpoint=True)
        nt.lfp_plot(axl, lt, laf, xticks[0], yticks_l[0], ns)
        axl.set_xlim(-0.1, 4.1)
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
        axp.set_xlim(-0.1, 4.1)
        axp.set_ylim([0, s_max])
        axp.tick_params(axis='y', which='minor', left=True, labelsize=4,
                        length=1.5,
                        width=0.3)
        axp.set_yticks(yticks_p[1], minor=True)
        axp.set_xticks(xticks[1], minor=True)
        axp.grid(which='both')

        for pt in np.array([0, 3.0, 4.0]):
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
c_fig.text(0.75, 0.05, 'Time (s)', ha='center', va='center', fontsize=15)
c_fig.text(0.27, 0.95, 'Averaged PSTH', ha='center', va='center', fontsize=15)
c_fig.text(0.75, 0.95, 'Averaged LFP', ha='center', va='center', fontsize=15)
c_fig.text(0.08, 0.5, 'Firing rate (spk/s)', ha='center', va='center',
           rotation='vertical', fontsize=15)
# c_fig.text(0.47, 0.5, 'Amplitude (mV)', ha='center', va='center',
#            rotation='vertical', fontsize=15)
c_fig.text(0.51, 0.5, 'Amplitude (mV)', ha='center', va='center',
           rotation='vertical', fontsize=15)
# c_fig.text(0.91, 0.5, 'Firing rate (spk/s)', ha='center', va='center',
#            rotation='vertical', fontsize=15)

for i in range(len(chs)):
    axs[i, 4].text(6, -400, depth[i] + ' $\mu$m',
                   ha='center', va='center', fontsize=10)
    if i == 0:
        axs[i, 4].text(6, 1000, 'Depth', ha='center', va='center', fontsize=12)

plt.savefig(snakemake.output[0], dpi=300)
# plt.savefig('testing', dpi=300)