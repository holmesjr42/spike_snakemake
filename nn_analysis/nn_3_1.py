import numpy as np
import spike_tools as st
import h5py
import nn_tool as nt
import parameters as pm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
import seaborn as sns

'''
This is for a single animal data
'''
file_path = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
            '/data/dataset/'

animal = 'cr50_190724'

d_lfp = file_path + 'lfp_fil/' + animal + '_lfp_fil.hdf5'
d_spk = file_path + 'tsp/' + animal + '_tsp.hdf5'

save_p = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
         '/dataset/image/nn_1_1/'
chs = [1, 3, 5, 7, 9, 11, 13, 14]

'''
setup
'''

sns.set_style('ticks')
val = pm.Setup(cond2='tsp', ch=chs)
cb = val.values()[-1]
depth = val.values()[6]
color = ['dodgerblue', 'tab:orange']
names = ['nn_3_1', 'nn_wn_3_1']

# lfp
ls = {}
la = {}
lr = []


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
xticks = np.array([np.linspace(0, 4, 5), np.arange(0, 4.1, 0.5)])
yticks_p = np.array([np.linspace(0, s_max, 3), np.linspace(0, s_max, 9)])
yticks_l = np.array([np.linspace(minf, maxf, 3), np.linspace(minf, maxf, 9)])


for i in range(len(chs)):
    p = chs[i]
    # axs[i, 1].axis('off')
    for j in range(len(names)):

        ns = names[j]
        axl = axs[i, 3*j+1]
        laf = la[ns][p]
        axp = axs[i, 3*j]

        # lfp
        lt = np.linspace(0, 4100, num=len(laf))
        nt.lfp_plot(axl, lt, laf, xticks[0], yticks_l[0], ns)
        axl.set_xlim(-0.1, 4.1)
        axl.set_ylim([minf, maxf])
        axl.set_yticks(yticks_l[1], minor=True)
        axl.tick_params(axis='y', which='minor', left=True, labelsize=4,
                        length=1.5,
                        width=0.3)
        axl.grid(which='both')
        axl.yaxis.tick_right()
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
        axp.grid(which='both')

        for pt in np.array([0, 3.0, 4.0]):
            axl.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.3)
            axp.axvline(x=pt, color='#031e57', linestyle='-.', lw=0.3)

        if i + 1 == len(chs):
            pass
        else:
            axl.xaxis.set_ticklabels([])
            axp.xaxis.set_ticklabels([])

    c_fig.delaxes(axs[i, 2])

'''
text and axis limit
'''

axs[0, 0].set_title('Averaged LFP')
axs[0, 1].set_title('Averaged PSTH')
axs[0, 3].set_title('Averaged LFP')
axs[0, 4].set_title('Averaged PSTH')
c_fig.text(0.27, 0.05, 'Time [s]', ha='center', va='center', fontsize=15)
c_fig.text(0.75, 0.05, 'Time [s]', ha='center', va='center', fontsize=15)
c_fig.text(0.27, 0.95, 'Notched Noise', ha='center', va='center', fontsize=15)
c_fig.text(0.75, 0.95, 'White Noise', ha='center', va='center', fontsize=15)
c_fig.text(0.52, 0.5, 'Firing rate [spk/s]', ha='center', va='center',
           rotation='vertical', fontsize=15)
c_fig.text(0.91, 0.5, 'Amplitude [mV]', ha='center', va='center',
           rotation='vertical', fontsize=15)
c_fig.text(0.08, 0.5, 'Firing rate [spk/s]', ha='center', va='center',
           rotation='vertical', fontsize=15)
c_fig.text(0.47, 0.5, 'Amplitude [mV]', ha='center', va='center',
           rotation='vertical', fontsize=15)
for i in range(len(chs)):
    axs[i, 4].text(7.6, -300, depth[i] + ' $\mu$m',
                   ha='center', va='center', fontsize=10)

plt.savefig('testing.png', dpi=300)
