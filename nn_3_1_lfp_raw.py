import spike_tools as st
import h5py
import nn_tool as nt
import parameters as pm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
import seaborn as sns
import numpy as np


'''
This is for a single animal data
'''
# file_path = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#             '/data/dataset/lfp_fil/'
# animal = 'cr50_190724_lfp_fil.hdf5'
# d_lfp = file_path + animal
d_lfp = snakemake.input[0]
save_p = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
         '/dataset/image/nn_3_1/'
chs = [1, 3, 5, 7, 9, 11, 13, 14]


# def nn_3_1(animal):
"""
setup
"""
sns.set_style('darkgrid')
val = pm.Setup(cond2='tsp', ch=chs)
depth = val.values()[6]
# cb = val.values()[-1]
color = ['dodgerblue', 'tab:orange']
ls = {}
la = {}
lr = []
names = ['nn_3_1', 'nn_wn_3_1']
maxs = []
mins = []
# yticks_p = np.array(
#     [np.arange(0, s_max + 1, s_max / 2), np.arange(0, s_max + 1, 50)])

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

        ls[n] = []
        la[n] = []
        for i in range(len(lst)):
            ls[n].append(lst[i][:-1])
            la[n].append(np.mean(lst[i][:-1], axis=0))
        maxa.append(np.max(la[n]))
        mina.append(np.min(la[n]))
        maxr.append(np.max(ls[n]))
        minr.append(np.min(ls[n]))
    maxs = np.append(maxs, [np.max(maxa), np.max(maxr)])
    mins = np.append(mins, [np.max(mina), np.max(minr)])

# print(maxs, mins)
'''
plot setup
'''
# min and max value for x, y limit
maxf = np.around(maxs, decimals=-2) + 100
minf = np.around(mins, decimals=-2) - 100
# print(maxf, minf)

# # lfp session overlapping figure
# l_fig_temp, axt = plt.subplots(len(chs), 2, figsize=(16, 9), dpi=300)
# # c_fig.subplots_adjust(bottom=0.7, hspace=0.3, wspace=0.07)
# l_fig_temp.patch.set_facecolor(cb)

# average lfp session figure
l_fig, axs = plt.subplots(len(chs), 5, figsize=(16, 9), dpi=300)
# c_fig.subplots_adjust(bottom=0.7, hspace=0.3, wspace=0.07)
# l_fig.patch.set_facecolor(cb)


for i in range(len(chs)):
    p = chs[i]
    # axs[i, 1].axis('off')
    for j in range(len(names)):
        ns = names[j]
        axl = [axs[i, j], axs[i, j+3]]
        lsf = ls[ns][p]
        laf = la[ns][p]
        for k in range(len(lsf)):
            t = np.linspace(0, 4100, num=len(lsf[k]))/1000
            axl[1].plot(t, lsf[k], alpha=0.3, lw=0.3)
            if k == 0:
                axl[0].plot(t, laf, lw=0.5, color=color[j])
            else:
                pass
        for lj in range(len(axl)):
            ax = axl[lj]
            ax.set_ylim(minf[lj], maxf[lj])
            ax.set_xlim(-0.1, 4.1)
        for ij in range(2):
            for jk in [0.0, 3.0, 4.0]:
                axl[ij].axvline(x=jk, color='#031e57', linestyle='-.', lw=0.3)

        if j == 0 and i != len(chs) - 1:
            axl[0].label_outer()
            axl[1].set_xticklabels([])
        elif j == 1:
            axl[0].label_outer()
            axl[1].label_outer()
        else:
            pass

        axl[0].set_yticks([-1000, 0.0, maxf[0]])

        # axp.set_yticks(yticks_p[1], minor=True)
        # axp.grid(which='both')
        # axp.axvline(x=3.0, color='#031e57', linestyle='-.', lw=0.3)
        # axp.axvline(x=4.0, color='#031e57', linestyle='-.', lw=0.3)

    l_fig.delaxes(axs[i, 2])

'''
text and axis limit
'''
axs[0, 0].set_title('NN On 3 s/ Off 1 s')
axs[0, 1].set_title('WN On 3 s/ Off 1 s')
axs[0, 3].set_title('NN On 3 s/ Off 1 s')
axs[0, 4].set_title('WN On 3 s/ Off 1 s')
l_fig.text(0.27, 0.05, 'Time [s]', ha='center', va='center', fontsize=15)
l_fig.text(0.75, 0.05, 'Time [s]', ha='center', va='center', fontsize=15)
l_fig.text(0.27, 0.95, 'Average', ha='center', va='center', fontsize=15)
l_fig.text(0.75, 0.95, 'All session', ha='center', va='center', fontsize=15)
l_fig.text(0.54, 0.5, 'Amplitude [mV]', ha='center', va='center',
           rotation='vertical', fontsize=15)
l_fig.text(0.06, 0.5, 'Amplitude [mV]', ha='center', va='center',
           rotation='vertical', fontsize=15)

for i in range(len(chs)):
    axs[i, 4].text(5, -300, depth[i] + ' $\mu$m',
                   ha='center', va='center', fontsize=10)


plt.savefig(save_p + d_lfp[-24:-13] + '_nn_3_1_lfp_s.png', dpi=300)
