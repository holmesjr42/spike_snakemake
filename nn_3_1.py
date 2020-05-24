# import glob
# import os.path as opath
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
setup
'''

sns.set_style('ticks')
val = pm.Setup(cond2='tsp', ch=[1, 3, 5, 7, 9, 11, 13, 14])
cb = val.values()[-1]
ch = [1, 3, 5, 7, 9, 11, 13, 14]  # channel selection
depth = val.values()[6]

file_path = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
            '/data/dataset/'

f = {}
lt = {}
sm = {}
sl = {}
nrastor = {}
npsth = {}
nsmooth = {}
ntime = {}
names = ['nn_3_1', 'nn_wn_3_1']

# yticks_p = np.array(
#     [np.arange(0, s_max + 1, s_max / 2), np.arange(0, s_max + 1, 50)])


'''
data retrieve
'''
# lfp
with h5py.File(file_path + 'cr33_190321' + '_lfp_fil.hdf5', 'r') as h_5:
    for n in names:
        lfp = h_5[n][0]
        ls = nt.lfp_seg(lfp, n)
        f[n], lt[n], sm[n], sl[n] = nt.m_spectro(ls)
        print(np.shape(f[n][0]), np.shape(lt[n]), np.shape(sl[n]))
        # yticks[n] = np.array([
        #                       np.arange(0, np.shape(f[n])[1], 20),
        #                       np.arange(0, np.shape(f[n])[1], 1)
        #                       ])

# spike
with h5py.File(file_path + 'cr33_190321' + '_tsp.hdf5', 'r') as h_5:
    for n in names:
        mua = h_5[n][0]
        nn = nt.index_spike(mua)
        nrastor[n], npsth[n], nsmooth[n], ntime[n] = nt.psth(nn, n)

s_max = np.around(np.max([
    np.max(nsmooth[names[0]]), np.max(nsmooth[names[1]])
    ]), decimals=-2) + 100
l_min = np.min(lt[names[0]])

'''
plot setup
'''
xticks_l = np.array([np.arange(0, 4.1, 1), np.arange(0, 4.1, 0.1)])
yticks_l = np.arange(0, 51, 10)
xticks_p = np.array([np.arange(0, 4.1, 1), np.arange(0, 4.1, 0.5)])
yticks_p = np.array([
    np.arange(0, s_max + 1, s_max / 2),
    np.arange(0, s_max + 1, 100)
    ])

c_fig, axs = plt.subplots(len(ch), 5, figsize=(16, 9), dpi=300)
c_fig.subplots_adjust(bottom=0.7, hspace=0.3, wspace=0.07)
c_fig.patch.set_facecolor(cb)
imag = []

for i in range(len(ch)):
    p = ch[i]
    # axs[i, 1].axis('off')
    for j in range(len(names)):
        k = names[j]
        axl = axs[i, j]
        nt.spect_plot(axl, lt[k][p], f[k][p], sl[k][p], imag, xticks_l,
                      yticks_l)
        axl.axvline(x=3.0, color='#031e57', linestyle='-.', lw=0.5)
        axl.axvline(x=4.0, color='#031e57', linestyle='-.', lw=0.5)
        axl.set_xlim(l_min, 4.1)
        axl.set_ylim(0, 51)

        axp = axs[i, j + 3]
        nt.psth_plot(axp, ntime[k], nsmooth[k][p], xticks_p[0], yticks_p[0],
                     s_max, k)
        axp.tick_params(axis='y', which='minor', left=True, labelsize=4,
                        length=1.5,
                        width=0.3)
        axp.set_yticks(yticks_p[1], minor=True)
        axp.grid(which='both')
        axp.axvline(x=3.0, color='#031e57', linestyle='-.', lw=0.3)
        axp.axvline(x=4.0, color='#031e57', linestyle='-.', lw=0.3)

        if j == 0:
            axl.label_outer()
        else:
            axl.label_outer()
            axp.label_outer()

    c_fig.delaxes(axs[i, 2])

'''
text and axis limit
'''
axs[0, 0].set_title('NN On 3 s/ Off 1 s')
axs[0, 1].set_title('WN On 3 s/ Off 1 s')
axs[0, 3].set_title('NN On 3 s/ Off 1 s')
axs[0, 4].set_title('WN On 3 s/ Off 1 s')
c_fig.text(0.28, 0.05, 'Time [s]', ha='center', va='center', fontsize=15)
c_fig.text(0.75, 0.05, 'Time [s]', ha='center', va='center', fontsize=15)
c_fig.text(0.55, 0.6, 'Firing rate [spk/s]', ha='center', va='center',
           rotation='vertical', fontsize=15)
c_fig.text(0.08, 0.6, 'Frequency [Hz]', ha='center', va='center',
           rotation='vertical', fontsize=15)

for i in range(len(ch)):
    axs[i, 1].text(-6.7, 30, depth[i] + '$\mu$m',
                   ha='center', va='center', fontsize=10)

# plt.setp(axs, xlim=(min(t['nn_3_1'][0]), 4), ylim=(0, 51))

'''
normalization
'''
vmins = np.around(min(image.get_array().min() for image in imag), decimals=2)
vmaxs = np.around(min(image.get_array().max() for image in imag), decimals=2)
c_fig.subplots_adjust(top=0.955, bottom=0.1)
norms = colors.Normalize(vmin=vmins, vmax=vmaxs)

for images in imag:
    images.set_norm(norms)

'''
colorbar
'''

# for i in range(len(ch)):
#     # to adjust the plot size for the first column
#     divider_off = make_axes_locatable(axs[i, 0])
#     cax_off = divider_off.append_axes("right", size="3%", pad=0.1)
#     cax_off.axis('off')
#
#     # real color bar
#
#     divider = make_axes_locatable(axs[i, 1])
#     cax = divider.append_axes("right", size="3%", pad=0.1)
#     cbar = c_fig.colorbar(imag[1], cax=cax, orientation='vertical',
#                           fraction=.01)
#     cbar.ax.tick_params(labelsize=8, width=0.1)
#     cbar.locator = ticker.MaxNLocator(nbins=5)
#     cbar.update_ticks()

[[x00, y00], [x01, y01]] = axs[0, 1].get_position().get_points()
[[x10, y10], [x11, y11]] = axs[7, 1].get_position().get_points()

cax = c_fig.add_axes([x01 + 0.02, y10, 0.01, y01 - y10])  # [bottom right x,
# bottom right y, width, height]

cbar = c_fig.colorbar(imag[1], cax=cax, orientation='vertical',
                      fraction=.01, pad=0.1)
cbar.ax.tick_params(labelsize=8, width=0.1)
cbar.locator = ticker.MaxNLocator(nbins=5)
cbar.update_ticks()

# plt.show()
plt.savefig('.png', dpi=300)
