import os
import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))
from tools import icsd
import numpy as np
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

'''
This is for a single data file.
CSD for nnm 3 s/ 1 s condition. Following traditional way with dividing 
with h**2 and no sigma is used. Uses rappel filter
'''

sns.set_style("ticks")
sr_lfp = 1061.5 * pq.Hz  # for convert time into data coord in the data file
comm = pm.Commonfeat
bfs = comm.bf_dic  # best frequency dictionary
b_list = comm.list_freq  # list of the 1/8 octave frequency to find index
# for bf/cf frequency
db = comm.db
lvl = db['lv']  # SPLdB list
la = {'nn': {}, 'si': {}}  # dict of lfp for nnm condition
ls = {}  # dict of lfp for strf condition
lw = {'nn': {}, 'si': {}}  # dict of lfp for wn condition
cond2 = ['nn', 'si']
csds = {}  # this is dict to store csd data

val = pm.Setup(cond='nnm_3_0', cond2='tsp', mode=1, ccmap='csd')
ccmap = val.values()[-1]
# ccmap = 'jet_r'
ws = val.values()[8]  # minor setup for plot
hs = 0.12

# file_input = snakemake.input[0]
file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
             '/data/dataset/nnm_lfp_avg/cr35_190403_lfp_avg_nnm.pk'
# file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#              '/data/dataset/nnm_lfp_avg/csd_lfp_avg_nnm.pk'

# getting animal cr number
animal = file_input.split('/')[-1].split('_')[0]

# loading file
with h5py.File(file_input, 'r') as dat:
    datas = dat

    if 'cr' in animal:
        if animal == 'cr30':
            plt.savefig(snakemake.output[0], dpi=300)
            sys.exit()
        else:
            pass

        bf = bfs[animal]
        print(bf)
        b_ind = np.where(b_list == bf)[0][0]

        # lfps  list
        lr = [datas['nnm_3_1'], datas['nnm_wn_3_1'], datas['nnm_3_1_strf']]

        for i in lvl:
            lra = [lr[0]['nn'][i], lr[0]['si'][i]]  # lfp for nnm
            lwa = [lr[1]['nn'][i], lr[1]['si'][i]]
            lrs = lr[2][i]  # lfp for strf
            for k in range(len(cond2)):
                la[cond2[k]][i] = []
                lw[cond2[k]][i] = []
            ls[i] = []

            for j in range(len(lrs)):
                ls[i].append(lrs[j][b_ind])
                for k in range(len(cond2)):
                    la[cond2[k]][i].append(lra[k][j][b_ind])
                    if np.any(lwa[k]):
                        lw[cond2[k]][i].append(lwa[k][j][b_ind])
                    else:
                        lw[cond2[k]][i].append(np.zeros(213))


        for lv in list(ls.keys()):
            csds[lv] = {}
            lfps = np.array([np.array(la['nn'][lv]), np.array(la['si'][lv]),
                             np.array(lw['nn'][lv]), np.array(lw['si'][lv]),
                             np.array(ls[lv])]
                            ) * pq.mV

            # print(la.keys())
            # 1, nn-on; 2, nn-off; 3, wn-on; 4, wn-off; 5, strf
            lfp_data1 = neo.core.AnalogSignal(lfps[0].T, sampling_rate=sr_lfp)
            lfp_data2 = neo.core.AnalogSignal(lfps[1].T, sampling_rate=sr_lfp)
            lfp_data3 = neo.core.AnalogSignal(lfps[2].T, sampling_rate=sr_lfp)
            lfp_data4 = neo.core.AnalogSignal(lfps[3].T, sampling_rate=sr_lfp)
            lfp_data5 = neo.core.AnalogSignal(lfps[4].T, sampling_rate=sr_lfp)

            coord = np.arange(0, 1501, 100) * pq.micrometer
            sigma = 1 / (100 * pq.micrometer)

            std_input_nnon = {
                'lfp': lfp_data1,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'gaussian',
                'f_order': (3, 1)
                }

            std_input_nnof = {
                'lfp': lfp_data2,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'gaussian',
                'f_order': (3, 1)
                }

            std_input_wnon = {
                'lfp': lfp_data3,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'gaussian',
                'f_order': (3, 1)
                }

            std_input_wnof = {
                'lfp': lfp_data4,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'gaussian',
                'f_order': (3, 1)
                }

            std_input_strf = {
                'lfp': lfp_data5,
                'coord_electrode': coord,
                'sigma': sigma,
                'f_type': 'gaussian',
                'f_order': (3, 1)
                }

            std1 = icsd.estimate_csd(**std_input_nnon)
            # rn_csd = np.transpose(std1[0] / std1[0].units) * std1[0].units
            f1_csd = np.transpose(std1[1] / std1[1].units) * std1[1].units
            # rn_csd.units = pq.mV / (pq.mm ** 2)
            f1_csd.units = pq.mV / (pq.mm ** 2)

            std2 = icsd.estimate_csd(**std_input_nnof)
            # rn_csd = np.transpose(std1[0] / std1[0].units) * std1[0].units
            f2_csd = np.transpose(std2[1] / std2[1].units) * std2[1].units
            # rn_csd.units = pq.mV / (pq.mm ** 2)
            f2_csd.units = pq.mV / (pq.mm ** 2)

            std3 = icsd.estimate_csd(**std_input_wnon)
            # rn_csd = np.transpose(std1[0] / std1[0].units) * std1[0].units
            f3_csd = np.transpose(std3[1] / std3[1].units) * std3[1].units
            # rn_csd.units = pq.mV / (pq.mm ** 2)
            f3_csd.units = pq.mV / (pq.mm ** 2)

            std4 = icsd.estimate_csd(**std_input_wnof)
            # rn_csd = np.transpose(std1[0] / std1[0].units) * std1[0].units
            f4_csd = np.transpose(std4[1] / std4[1].units) * std4[1].units
            # rn_csd.units = pq.mV / (pq.mm ** 2)
            f4_csd.units = pq.mV / (pq.mm ** 2)

            std5 = icsd.estimate_csd(**std_input_strf)
            # rw_csd = np.transpose(std2[0] / std2[0].units) * std2[0].units
            f5_csd = np.transpose(std5[1] / std5[1].units) * std5[1].units
            # rw_csd.units = pq.mV / (pq.mm ** 2)
            f5_csd.units = pq.mV / (pq.mm ** 2)

            csds[lv] = [f1_csd, f2_csd, f3_csd, f4_csd, f5_csd]

    elif 'csd' in animal:

        if '30' in list(datas['lfp']['nnm_3_1_strf'].keys()):
            datas['lfp']['nnm_3_1_strf'].pop('30')
            datas['csd']['nnm_3_1_strf'].pop('30')
        else:
            pass

        if '30' in list(datas['lfp']['nnm_3_1'].keys()):
            datas['lfp']['nnm_3_1'].pop('30')
            datas['csd']['nnm_3_1'].pop('30')
        else:
            pass

        if '30' in list(datas['lfp']['nnm_wn_3_1'].keys()):
            datas['lfp']['nnm_wn_3_1'].pop('30')
            datas['csd']['nnm_wn_3_1'].pop('30')
        else:
            pass

        for lv in lvl:
            la['nn'][lv] = {}
            la['si'][lv] = {}
            lw['nn'][lv] = {}
            lw['si'][lv] = {}
            ls[lv] = {}
            csds[lv] = {}
            csds[lv] = [np.array(datas['csd']['nnm_3_1'][lv][0]) *
                        pq.mV / (pq.mm ** 2),
                        np.array(datas['csd']['nnm_3_1'][lv][1]) *
                        pq.mV / (pq.mm ** 2),
                        np.array(datas['csd']['nnm_wn_3_1'][lv][0]) *
                        pq.mV / (pq.mm ** 2),
                        np.array(datas['csd']['nnm_wn_3_1'][lv][1]) *
                        pq.mV / (pq.mm ** 2),
                        np.array(datas['csd']['nnm_3_1_strf'][lv]) *
                        pq.mV / (pq.mm ** 2)]

            for k in range(len(cond2)):
                la[cond2[k]][lv] = np.array(datas['lfp']['nnm_3_1'][lv][k]) * \
                                   pq.mV
                lw[cond2[k]][lv] = np.array(
                    datas['lfp']['nnm_wn_3_1'][lv][k]) * pq.mV
            ls[lv] = np.array(datas['lfp']['nnm_3_1_strf'][lv]) * pq.mV


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

    ax.tick_params(axis='x', which='major', labelsize=6)
    ax.tick_params(axis='y', which='major', labelsize=6)

    return ax, axins


def cbar_t(cbar, type):
    cbar.ax.tick_params(labelsize=7)
    if type == 'lfp':
        cbar.set_label(
            r'mV',
            labelpad=15, size=7
            )

    elif type == 'csd':
        cbar.set_label(
            r'$\times$10$^{3}$' + '\n ' + r'mV/$mm^{2}$',
            labelpad=15, size=7
            )


'''
plots
'''
# setup
c_img = []
l_img = []
ax_ind = []
axisl = []
axisc = []
axinnl = []
axinfl = []
axiwnl = []
axiwfl = []
axinnc = []
axinfc = []
axiwnc = []
axiwfc = []

lvp = []  # spl list which actually has data
for i in range(len(lvl)):

    if not np.any(la['nn'][lvl[i]]):
        continue
    else:
        lvp.append(lvl[i])

c_fig = plt.figure(0, figsize=(16, 9), dpi=300)

# st_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
#                                                        top=0.95, bottom=0.8,
#                                                        wspace=ws, hspace=hs))
#
# nn_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
#                                                        top=0.77, bottom=0.62,
#                                                        wspace=ws, hspace=hs))
#
# nf_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
#                                                        top=0.59, bottom=0.44,
#                                                        wspace=ws, hspace=hs))
#
# wn_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
#                                                        top=0.41, bottom=0.26,
#                                                        wspace=ws, hspace=hs))
#
# wf_axes = c_fig.subplots(2, len(lvp), gridspec_kw=dict(left=0.05,
#                                                        top=0.23, bottom=0.08,
#                                                        wspace=ws, hspace=hs))

axes_70 = c_fig.subplots(3, 2,
                         gridspec_kw=dict(left=0.05, right=0.21,
                                          top=0.95, bottom=0.1,
                                          wspace=0.001, hspace=hs))

axes_60 = c_fig.subplots(3, 2,
                         gridspec_kw=dict(left=0.26, right=0.42,
                                          top=0.95, bottom=0.1,
                                          wspace=ws, hspace=hs))

axes_50 = c_fig.subplots(3, 2,
                         gridspec_kw=dict(left=0.47, right=0.63,
                                          top=0.95, bottom=0.1,
                                          wspace=ws, hspace=hs))

axes_40 = c_fig.subplots(3, 2,
                         gridspec_kw=dict(left=0.68, right=0.84,
                                          top=0.95, bottom=0.1,
                                          wspace=ws, hspace=hs))


for i in range(len(lvp)):
    nn_lfp = la['nn'][lvp[i]] * pq.mV
    nf_lfp = la['si'][lvp[i]] * pq.mV
    wn_lfp = lw['nn'][lvp[i]] * pq.mV
    wf_lfp = lw['si'][lvp[i]] * pq.mV
    s_lfp = ls[lvp[i]] * pq.mV

    nn_csd = csds[lvp[i]][0]
    nf_csd = csds[lvp[i]][1]
    wn_csd = csds[lvp[i]][2]
    wf_csd = csds[lvp[i]][3]
    s_csd = csds[lvp[i]][4]

    ax_nnl, axins_nnl = csd_plot(zz=nn_lfp, ax=nn_axes[2 * i, 0], image=l_img)
    ax_nfl, axins_nfl = csd_plot(zz=nf_lfp, ax=nn_axes[2 * i, 1], image=l_img)
    ax_wnl, axins_wnl = csd_plot(zz=wn_lfp, ax=wn_axes[2 * i, 0], image=l_img)
    ax_wfl, axins_wfl = csd_plot(zz=wf_lfp, ax=wn_axes[2 * i, 1], image=l_img)
    ax_sl, axins_sl = csd_plot(zz=s_lfp, ax=st_axes[2 * i, 1], image=l_img)

    ax_nnc, axins_nnc = csd_plot(zz=nn_csd, ax=nn_axes[2 * i + 1, 0],
                                 image=c_img)
    ax_nfc, axins_nfc = csd_plot(zz=nf_csd, ax=nn_axes[2 * i + 1, 1],
                                 image=c_img)
    ax_wnc, axins_wnc = csd_plot(zz=wn_csd, ax=wn_axes[2 * i + 1, 0],
                                 image=c_img)
    ax_wfc, axins_wfc = csd_plot(zz=wf_csd, ax=wn_axes[2 * i + 1, 1],
                                 image=c_img)
    ax_sc, axins_sc = csd_plot(zz=s_csd, ax=st_axes[2 * i + 1, 1], image=c_img)

    ax_ind = np.append(ax_ind, i)
    axisl.append(axins_sl)
    axisc.append(axins_sc)
    axinnl.append(axins_nnl)
    axinnc.append(axins_nnc)
    axinfl.append(axins_nfl)
    axinfc.append(axins_nfc)
    axiwnl.append(axins_wnl)
    axiwnc.append(axins_wnc)
    axiwfl.append(axins_wfl)
    axiwfc.append(axins_wfc)

    st_axes[2 * i, 0].set_visible(False)
    st_axes[2 * i + 1, 0].set_axis_off()

    st_axes[2 * i, 1].set_ylabel('Depth' + ' (' + r'$\mu$' + 'm' + ')',
                                 labelpad=10, fontdict={'size': 9}
                                 )
    st_axes[2 * i + 1, 1].set_ylabel('Depth' + ' (' + r'$\mu$' + 'm' + ')',
                                     labelpad=10, fontdict={'size': 9}
                                     )

for i in range(2 * len(lvp) - 1):
    st_axes[i, 1].get_xaxis().set_ticks([])
    nn_axes[i, 0].get_xaxis().set_ticks([])
    nn_axes[i, 1].get_xaxis().set_ticks([])
    wn_axes[i, 0].get_xaxis().set_ticks([])
    wn_axes[i, 1].get_xaxis().set_ticks([])

st_axes[0, 1].set_title(f'STRF\nCONTROL', pad=10,
                        fontdict={'weight': 'bold',
                                  'size': 8
                                  })
nn_axes[0, 0].set_title(f'NN-ON', pad=10,
                        fontdict={'weight': 'bold',
                                  'size': 8
                                  })
nn_axes[0, 1].set_title(f'NN-OFF', pad=10,
                        fontdict={'weight': 'bold',
                                  'size': 8
                                  })
wn_axes[0, 0].set_title(f'WN-ON', pad=10,
                        fontdict={'weight': 'bold',
                                  'size': 8
                                  })
wn_axes[0, 1].set_title(f'WN-OFF', pad=10,
                        fontdict={'weight': 'bold',
                                  'size': 8
                                  })
xs = len(lvp) - 1
st_axes[xs, 1].set_xlabel('t (ms)', labelpad=10, fontdict={'size': 9})
nn_axes[xs, 0].set_xlabel('t (ms)', labelpad=10, fontdict={'size': 9})
nn_axes[xs, 1].set_xlabel('t (ms)', labelpad=10, fontdict={'size': 9})
wn_axes[xs, 0].set_xlabel('t (ms)', labelpad=10, fontdict={'size': 9})
wn_axes[xs, 1].set_xlabel('t (ms)', labelpad=10, fontdict={'size': 9})

for i in range(len(lvp)):
    st_axes[2 * i, 1].text(-1.5, 0, f'{lvp[i]} dB SPL',
                           horizontalalignment='center',
                           verticalalignment='center',
                           rotation='vertical',
                           transform=st_axes[2 * i, 1].transAxes,
                           fontdict={'weight': 'bold', 'size': 11}
                           )


def norm_color(img, type, ratio=0.7):
    if type == 'lfp':
        # vmin = min(ims.get_array().min() for ims in img) * ratio
        # vmax = max(ims.get_array().max() for ims in img) * ratio
        vmax = max(max(ims.get_array().max() for ims in img), np.abs(min(
            ims.get_array().min() for ims in img))) * ratio
        vmin = (-1) * vmax
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
    cbar_sl = plt.colorbar(l_img[i], cax=axisl[i],
                           orientation='vertical', format='%.2f',
                           ticks=ticker.MaxNLocator(5))
    cbar_sc = plt.colorbar(c_img[i], cax=axisc[i],
                           orientation='vertical', format='%.2f',
                           ticks=ticker.MaxNLocator(5))

    cbar_nnl = plt.colorbar(l_img[i], cax=axinnl[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))
    cbar_nnc = plt.colorbar(c_img[i], cax=axinnc[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))
    cbar_nfl = plt.colorbar(l_img[i], cax=axinfl[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))
    cbar_nfc = plt.colorbar(c_img[i], cax=axinfc[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))

    cbar_wnl = plt.colorbar(l_img[i], cax=axiwnl[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))
    cbar_wnc = plt.colorbar(c_img[i], cax=axiwnc[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))
    cbar_wfl = plt.colorbar(l_img[i], cax=axiwfl[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))
    cbar_wfc = plt.colorbar(c_img[i], cax=axiwfc[i],
                            orientation='vertical', format='%.2f',
                            ticks=ticker.MaxNLocator(5))

    cbar_t(cbar_wfl, 'lfp')
    cbar_t(cbar_wfc, 'csd')

    axisl[i].remove()
    axisc[i].remove()
    axinnl[i].remove()
    axinnc[i].remove()
    axinfl[i].remove()
    axinfc[i].remove()
    axiwnl[i].remove()
    axiwnc[i].remove()
#
# # plt.show()
c_fig.savefig('csd_test', dpi=300)
# c_fig.savefig(snakemake.output[0], dpi=300)
