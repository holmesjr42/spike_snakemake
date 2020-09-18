import os,sys,inspect, pathlib
sys.path.insert(1, os.path.dirname(os.getcwd()))
import numpy as np
from tools import icsd
from parameters import parameters as pm
import pickle
import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
import seaborn as sns
import quantities as pq
import neo.core
sys.path.pop(1)

'''This is for a single data file. CSD for nnm continuous condition. Best 
frequeny at the high edge of the gap. Following traditional way with 
dividing with h**2 and no sigma is used. Uses rappel filter '''

sns.set_style("ticks")
sr_lfp = 1061.5 * pq.Hz  # for convert time into data coord in the data file
comm = pm.Commonfeat
bfs = comm.bf_dic  # best frequency dictionary
b_list = comm.list_freq  # list of the 1/8 octave frequency to find index
# for bf/cf frequency
db = comm.db
lvl = db['lv']  # SPLdB list
la = {}  # dict of lfp for nnm condition
ls = {}  # dict of lfp for strf condition
csds = {}  # this is dict to store csd data
val = pm.Setup(cond='nnm_3_0', cond2='tsp', mode=1, ccmap='csd')
ccmap = val.values()[-1]
# ccmap = 'jet_r'
ws = val.values()[8]  # minor setup for plot
hs = 0.05

file_input = snakemake.input[0]
# file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#              '/data/dataset/nnm_lfp_avg/cr50_190724_lfp_avg_nnm.pk'
# file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#              '/data/dataset/nnm_lfp_avg/csd_lfp_avg_nnm.pk'

# getting animal cr number
animal = file_input.split('/')[-1].split('_')[0]

# loading data
with h5py.File(file_input, 'r') as dat:
    datas = dat

    if 'cr' in animal:
        bf = bfs[animal]
        cf = np.round(bf * 2**(-0.5))

        if cf == 9513:
            cf = 9514
        elif cf == 3999:
            cf = 4000
        else:
            pass

        b_ind = np.where(b_list == cf)[0][0]
        print(b_ind)

        lr = [datas['nnm_hef'], datas['nnm_hef_strf']]  # lfps list

        for i in lvl:
            lra = np.array(lr[0][i])  # lfp for nnm
            lrs = np.array(lr[1][i])  # lfp for strf


            la[i] = []
            ls[i] = []
            for j in range(len(lrs)):
                la[i].append(lra[j][b_ind])
                ls[i].append(lrs[j][b_ind])



        for lv in list(la.keys()):
            csds[lv] = []
            lfps = np.array([la[lv], ls[lv]]) * pq.mV
            # print(la.keys())
            lfp_data1 = neo.core.AnalogSignal(lfps[0].T, sampling_rate=sr_lfp)
            lfp_data2 = neo.core.AnalogSignal(lfps[1].T, sampling_rate=sr_lfp)
            coord = np.arange(0, 1501, 100) * pq.micrometer
            sigma = 1 / (100 * pq.micrometer)

            std_input_nnm = {
                'lfp': lfp_data1,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'rappel',
                'f_order': 3,
                }

            std_input_strf = {
                'lfp': lfp_data2,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'rappel',
                'f_order': 3,
                }

            std1 = icsd.estimate_csd(**std_input_nnm)
            # rn_csd = np.transpose(std1[0] / std1[0].units) * std1[0].units
            fn_csd = np.transpose(std1[1] / std1[1].units) * std1[1].units
            # rn_csd.units = pq.mV / (pq.mm ** 2)
            fn_csd.units = pq.mV / (pq.mm ** 2)

            std2 = icsd.estimate_csd(**std_input_strf)
            # rw_csd = np.transpose(std2[0] / std2[0].units) * std2[0].units
            fs_csd = np.transpose(std2[1] / std2[1].units) * std2[1].units
            # rw_csd.units = pq.mV / (pq.mm ** 2)
            fs_csd.units = pq.mV / (pq.mm ** 2)

            csds[lv].append(fn_csd)
            csds[lv].append(fs_csd)

    elif 'csd' in animal:
        if '30' in list(datas['lfp']['nnm_hef_strf'].keys()):
            datas['lfp']['nnm_hef_strf'].pop('30')
            datas['csd']['nnm_hef_strf'].pop('30')
        else:
            pass

        if '30' in list(datas['lfp']['nnm_hef'].keys()):
            datas['lfp']['nnm_hef'].pop('30')
            datas['csd']['nnm_hef'].pop('30')
        else:
            pass

        for lv in lvl:
            la[lv] = {}
            ls[lv] = {}
            csds[lv] = {}
            csds[lv] = [np.array(datas['csd']['nnm_hef'][lv]) *
                        pq.mV / (pq.mm ** 2),
                        np.array(datas['csd']['nnm_hef_strf'][lv]) *
                        pq.mV / (pq.mm ** 2)]
            la[lv] = np.array(datas['lfp']['nnm_hef'][lv]) * pq.mV
            ls[lv] = np.array(datas['lfp']['nnm_hef_strf'][lv]) * pq.mV
'''
plot setup
'''


def csd_plot(zz, ax, image, cmap=ccmap):
    zz = zz / zz.units
    sig = zz * 10 ** (-3)
    t_ticks = np.linspace(0, 200, 9, endpoint=True)
    t_ticks2 = np.linspace(0, 200, 5, endpoint=True)
    img = ax.imshow(
        sig, cmap=cmap, aspect=213 / 1500,
        extent=[0, 200, 1500, 0], interpolation='gaussian')
    # aspect: the number of elements in x-axis
    image.append(img)
    yticks = np.arange(0, 1501, 100)
    yticks2 = np.array([0, 500, 800, 1100, 1400])
    ax.set_yticks(yticks2)
    ax.set_yticks(yticks, minor=True)
    ax.set_xticks(t_ticks, minor=True)
    ax.set_xticks(t_ticks2)

    axins = inset_axes(ax,
                       width="5%",  # width = 5% of parent_bbox width
                       height="100%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.1, 0., 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )

    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.tick_params(axis='y', which='major', labelsize=9)

    return ax, axins


def cbar_t(cbar, type):
    cbar.ax.tick_params(labelsize=7)
    if type == 'lfp':
        cbar.set_label(
            r'mV',
            labelpad=15, size=9
            )

    elif type == 'csd':
        cbar.set_label(
            r'$\times$10$^{-3}$' + ' ' + r'mV/$mm^{2}$',
            labelpad=15, size=9
            )


'''
plots
'''
# setup
c_img = []
l_img = []
ax_ind = []
axinsl = []
axinsc = []
axinnl = []
axinnc = []

lvp = []  # spl list which actually has data
for i in range(len(lvl)):
    if not np.any(la[lvl[i]]):
        continue
    else:
        lvp.append(lvl[i])

c_fig = plt.figure(0, figsize=(16, 9), dpi=300)

st_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
                                                       top=0.95, bottom=0.55,
                                                       wspace=ws, hspace=hs))

on_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
                                                       top=0.5, bottom=0.1,
                                                       wspace=ws, hspace=hs))

for i in range(len(lvp)):

    n_lfp = la[lvp[i]] * pq.mV
    s_lfp = ls[lvp[i]] * pq.mV
    n_csd = csds[lvp[i]][0]
    s_csd = csds[lvp[i]][1]

    ax_sl, axins_sl = csd_plot(zz=s_lfp, ax=st_axes[0, i], image=l_img)
    ax_sc, axins_sc = csd_plot(zz=s_csd, ax=st_axes[1, i], image=c_img)
    ax_nl, axins_nl = csd_plot(zz=n_lfp, ax=on_axes[0, i], image=l_img)
    ax_nc, axins_nc = csd_plot(zz=n_csd, ax=on_axes[1, i], image=c_img)
    ax_ind = np.append(ax_ind, i)
    axinsl.append(axins_sl)
    axinsc.append(axins_sc)
    axinnl.append(axins_nl)
    axinnc.append(axins_nc)

    st_axes[0, i].set_title(f'{lvp[i]} dB SPL', pad=13,
                            fontdict={'weight': 'bold',
                                      'size': 11
                                      })

    on_axes[1, i].set_xlabel('t (ms)', labelpad=10)
    for j in range(2):
        if i == 0:
            st_axes[j, i].set_ylabel('Depth' + ' (' + r'$\mu$' + 'm' + ')',
                                     labelpad=10
                                     )
            on_axes[j, i].set_ylabel('Depth' + ' (' + r'$\mu$' + 'm' + ')',
                                     labelpad=10
                                     )


    st_axes[0, i].get_xaxis().set_ticks([])
    on_axes[0, i].get_xaxis().set_ticks([])

st_axes[0, 0].text(-0.8, 0, 'STRF\nControl',
                   horizontalalignment='center',
                   verticalalignment='center',
                   rotation='vertical',
                   transform=st_axes[0, 0].transAxes,
                   fontdict={'weight': 'bold', 'size': 11}
                   )

on_axes[0, 0].text(-0.8, 0, 'NN\nContinuous',
                   horizontalalignment='center',
                   verticalalignment='center',
                   rotation='vertical',
                   transform=on_axes[0, 0].transAxes,
                   fontdict={'weight': 'bold', 'size': 11}
                   )

def norm_color(img, type, ratio=0.7):
    if type == 'lfp':
        vmin = min(ims.get_array().min() for ims in img) * ratio
        vmax = max(ims.get_array().max() for ims in img) * ratio
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif type == 'csd':
        vmax = max(max(ims.get_array().max() for ims in img), np.abs(min(
            ims.get_array().min() for ims in img))) * ratio
        vmin = (-1) * vmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        print('there is a problem with norm_color. check it')
        sys.exit()

    for ims in img:
        ims.set_norm(norm)


norm_color(l_img, 'lfp')
norm_color(c_img, 'csd')

for i in range(len(ax_ind)):

    cbar_sl = plt.colorbar(l_img[i], cax=axinsl[i],
                           orientation='vertical', format='%.2f',
                           ticks=ticker.MaxNLocator(5))
    cbar_sc = plt.colorbar(c_img[i], cax=axinsc[i],
                           orientation='vertical', format='%.2f',
                           ticks=ticker.MaxNLocator(5))
    cbar_nl = plt.colorbar(l_img[i], cax=axinnl[i],
                           orientation='vertical', format='%.2f',
                           ticks=ticker.MaxNLocator(5))
    cbar_nc = plt.colorbar(c_img[i], cax=axinnc[i],
                           orientation='vertical', format='%.2f',
                           ticks=ticker.MaxNLocator(5))

    if ax_ind[i] == ax_ind[-1]:

        cbar_t(cbar_sl, 'lfp')
        cbar_t(cbar_sc, 'csd')
        cbar_t(cbar_nl, 'lfp')
        cbar_t(cbar_nc, 'csd')

    else:
        axinsl[i].remove()
        axinsc[i].remove()
        axinnl[i].remove()
        axinnc[i].remove()

# plt.show()
# c_fig.savefig('csd_test', dpi=300)
c_fig.savefig(snakemake.output[0], dpi=300)
