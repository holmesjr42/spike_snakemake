import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
from parameters import parameters as pm
import seaborn as sns
import h5py
from matplotlib.colors import ListedColormap
from tools import nnm_plot as nnp
import quantities as pq
sys.path.pop(1)

'''
this is to process and generate a single data file for spikes and lfp 
data
'''
# sns.set_style("ticks")

# import features
nnc = pm.Nnmcontfeat
nns = pm.Nnsifeat
com = pm.Commonfeat

# some adjustments for plots: channels you are going to use, wspace, hspace,
# colormap, min and max values for heatmap

# continuous nn
val = pm.Setup(cond='nnm_3_0', cond2='tsp', mode=1)
val2 = pm.Setup(cond='nnm_3_0', cond2='lfp', mode=1, ccmap='lfp')
mode, ylc, condc, condt, nb, ch, depth, dcmap, ws, hs, rt, cct, \
cb, ccmap = val.values()
ccl = val2.values()[-3]  # color for notch borders
# lfp heatmap colormap
lcmap = val2.values()[-1]
condl = 'lfp'
octs = com.octaves  # octave range for plotting
lvl = com.db['lv']  # SPLdB list
g_n = {}  # dict of lfp for nnm condition
g_s = {}  # dict of lfp for strf condition
# dataset
file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
             '/data/dataset/nnm_spk_lfp_avg/all_spk_lfp_avg_nnm.hdf5'
# file_input = snakemake.input[0]
# getting animal cr number
animal = nnp.animal_name(file_input)

# conditions for this script
c_n = 'nnm_3_0'
c_s = 'nnm_3_0_strf'

# loading data
with h5py.File(file_input, 'r') as dat:
    datas = nnp.pop_lvl(animal, dat, [c_n, c_s])

    # aranging dataset

    sk = [datas['tsp'][c_n], datas['tsp'][c_s]]  # spk
    lp = [datas['lfp'][c_n], datas['lfp'][c_s]]  # lfp

    sa = [datas['fpt'][c_n], datas['fpt'][c_s]]  # spk_average (f profile)
    sc = [datas['scf'][c_n], datas['scf'][c_s]]  # spk_average confidence
    # interval

    la = [datas['fpl'][c_n], datas['fpl'][c_s]]  # lfp_average (f profile)
    lc = [datas['lcf'][c_n], datas['lcf'][c_s]]  # lfp_average confidence
    # interval

    for i in lvl:
        g_n[i] = [np.array(sk[0][i]), np.array(lp[0][i]),
                  np.array(sa[0][i]), np.array(la[0][i]),
                  np.array(sc[0][i]), np.array(lc[0][i])
                  ]
        g_s[i] = [np.array(sk[1][i]), np.array(lp[1][i]),
                  np.array(sa[1][i]), np.array(la[1][i]),
                  np.array(sc[1][i]), np.array(lc[1][i])
                  ]

# plot setup
'''
nn continuous
'''

#  heatmap
# setup
c_img = []

lvp = []  # spl list which actually has data or choose spl you want to use
for i in range(len(lvl)):
    if not np.any(g_n[lvl[i]][0]):
        continue
    else:
        lvp.append(lvl[i])

lv_off = [x for x in lvl if x not in lvp]  # this is to remove the column
# for non-exist SPL data from certain animals
#
# """spike heatmap"""
#
# c1_fig = plt.figure(0, figsize=(16, 9), dpi=300)
# c1_axs1 = c1_axs2 = c1_axs3 = c1_axs4 = 0
# c_axes = [c1_axs1, c1_axs2, c1_axs3, c1_axs4]
#
# for i in range(len(lvp)):
#     c_axes[i] = nnp.sub_plots(c1_fig, len(ch), 2, i, ws, hs)
#
# # dic for axis
# c1_axs = {}
# for i in range(len(lvp)):
#     c1_axs[lvp[i]] = c_axes[i]
#
# # fig.suptitle('Multiple images')
#
# for j in lvp:
#     for i in range(len(ch)):
#         tspn = g_n[j][0]
#         tspc = g_s[j][0]
#         aaxn = c1_axs[j][i, 1]
#         aaxc = c1_axs[j][i, 0]
#
#         nnp.nnm_plot(aaxn, tspn[ch[i]], c_img, cm=ccmap)
#         nnp.nnm_plot(aaxc, tspc[ch[i]], c_img, cm=ccmap)
#
#         aaxs = [aaxn, aaxc]
#         # notch indication and label texts
#         nnp.nnm_labels(aaxs, 0.5, lvp, depth, ch, j, i)
#
# for i in lv_off:
#     for j in range(len(ch)):
#         c1_axs[i][j, 0].set_axis_off()
#         c1_axs[i][j, 1].set_axis_off()
#
# # heatmap normalization
# cmx = nnp.norm_color(c_img, 'tsp', rt)
# # colorbar
# for i in range(len(lvp)):
#     nnp.colorbar_set(c1_fig, c1_axs, cmx, lvp[i], lvp[0], c_img[i], 'tsp')
# # plt.savefig('test0.png', dpi=300)
# plt.savefig(snakemake.output[0], dpi=300)
# c1_fig.clf()
#
# """lfp heatmap"""
#
# l_img = []
#
# l1_fig = plt.figure(0, figsize=(16, 9), dpi=300)
#
# l1_axs1 = l1_axs2 = l1_axs3 = l1_axs4 = 0
# l_axes = [l1_axs1, l1_axs2, l1_axs3, l1_axs4]
#
# for i in range(len(lvp)):
#     l_axes[i] = nnp.sub_plots(l1_fig, len(ch), 2, i, ws, hs)
#
#
# # dic for axis
# l1_axs = {}
# for i in range(len(lvp)):
#     l1_axs[lvp[i]] = l_axes[i]
#
# for j in lvp:
#     for i in range(len(ch)):
#         tspn = g_n[j][1] * pq.microvolt
#         tspn.units = pq.mV
#         tspc = g_s[j][1] * pq.microvolt
#         tspc.units = pq.mV
#
#         aaxn = l1_axs[j][i, 1]
#         aaxc = l1_axs[j][i, 0]
#         # print('hey', tspc[i])
#         nnp.nnm_plot(aaxn, tspn[ch[i]], l_img, cm=lcmap)
#         nnp.nnm_plot(aaxc, tspc[ch[i]], l_img, cm=lcmap)
#         # axinc.append(axin_s)
#
#         aaxs = [aaxn, aaxc]
#
#         # notch indication and label texts
#         nnp.nnm_labels(aaxs, 0.5, lvp, depth, ch, j, i)
#
#
# for i in lv_off:
#     for j in range(len(ch)):
#         l1_axs[i][j, 0].set_axis_off()
#         l1_axs[i][j, 1].set_axis_off()
#
# # heatmap normalization
# lmx = nnp.norm_color(l_img, 'lfp', rt)
#
# # colorbar
# for i in range(len(lvp)):
#     nnp.colorbar_set(l1_fig, l1_axs, lmx, lvp[i], lvp[0], l_img[i], 'lfp')
#
# # plt.show()
# plt.savefig('test1.png', dpi=300)
# # plt.savefig(snakemake.output[1], dpi=300)
# plt.clf()
#
# frequency profile
f1_fig = plt.figure(figsize=(16, 9), dpi=300)
f1_fig.subplots_adjust(left=0.12, right=0.92, top=0.9, bottom=0.1)
f1_axs = f1_fig.subplots(len(ch), 4, gridspec_kw=dict(wspace=0.15, hspace=0.2))

f_max = []
fmd = 0

for j in range(len(lvp)):
    pi = lvl.index(lvp[j])
    for i in range(len(ch)):
        tspn = g_n[lvp[j]][2][ch[i]]
        tspc = g_s[lvp[j]][2][ch[i]]
        # tvn = g_n[lvp[j]][6][ch[i]]
        # tvc = g_s[lvp[j]][6][ch[i]]
        #
        f_max.append([np.max(tspn), np.max(tspc)])

ym = np.around(np.max(f_max) * 1.3, decimals=2)

with open('/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data'
          '/dataset/wilcoxon/nnm_3_0_wilcoxon_f.pk', 'rb') as datas:
    impf = pickle.load(datas)

with open('/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data'
          '/dataset/wilcoxon/nnm_3_0_wilcoxon_l.pk', 'rb') as datas:
    impl = pickle.load(datas)

for j in lvp:
    pi = lvl.index(j)
    for i in range(len(ch)):
        fa = f1_axs[i, pi]
        tspn = g_n[j][2]
        tspc = g_s[j][2]
        tvn = g_n[j][6]
        tvc = g_s[j][6]

        p_dat = [tspc, tspn]
        p_var = [tvc, tvn]
        labelz = ['Control', 'NN']
        ltype = ['-', '--']

        for k in range(len(p_dat)):
            fa.plot(octs, p_dat[k][ch[i]],
                    label=labelz[k], lw=1,
                    color=dcmap[k], linestyle=ltype[k]
                    )

            if 'all' in animal:
                pass
            else:
                fa.fill_between(
                    octs,
                    p_var[k][ch[i]][:, 0],
                    p_var[k][ch[i]][:, 1],
                    alpha=.2
                    )

        fa.set_xticks([-1.5, -1., -0.5, 0, 0.5, 1., 1.5])
        fa.set_yticks(np.linspace(0, ym, 3, endpoint=True))
        fa.set_ylim(top=ym, bottom=0)
        fa.set_xlim(left=-1.5, right=1.5)
        fa.tick_params(axis='both', which='major', labelsize=8, length=3,
                       width=1)
        fa.grid(True, alpha=0.6, lw=0.5)

        for p in [-0.5, 0.5]:
            fa.axvline(x=p, color='#363b50', linestyle='-.', lw=0.7)

        # significance
        if impf[j][ch[i]] <= .05:
            fa.scatter(0, tspn[ch[i]][12] + 0.03, marker=(5, 2), c='black',
                       alpha=0.7)
        else:
            pass
        # tick lable off
        if i != len(ch) - 1:
            fa.tick_params(labelbottom=False)

        if j == lvp[0]:
            fa.set_ylabel(depth[i] + r'$\mu$' + 'm' + '\n\n\n\n\n',
                          fontsize=10)
        else:
            fa.tick_params(labelleft=False)

        # legend box
        if j == lvp[len(lvp) - 1] and i == 0:
            fa.legend(prop={'size': 7.5},
                      loc='upper right',
                      bbox_to_anchor=(1, 1))
        else:
            pass

        if i == 0:
            fa.text(0.5, 1.5, f'{j} SPL dB',
                    horizontalalignment='center',
                    verticalalignment='center',
                    # rotation='vertical',
                    transform=fa.transAxes,
                    fontdict={'weight': 'bold', 'size': 11}
                    )
        elif i == len(ch) - 1:
            fa.text(0.5, -0.3, f'Octave',
                    horizontalalignment='center',
                    verticalalignment='center',
                    # rotation='vertical',
                    transform=fa.transAxes,
                    fontdict={'weight': 'bold', 'size': 11}
                    )

f1_axs[1, 0].text(-0.4, -0.7, 'Depth\n\n\n\n\n\nFiring rate (spk/s)',
                  horizontalalignment='center',
                  verticalalignment='center',
                  rotation='vertical',
                  transform=f1_axs[1, 0].transAxes,
                  fontdict={'weight': 'bold', 'size': 11}
                  )

for j in lv_off:
    o_i = lvl.index(j)
    for i in range(len(ch)):
        f1_axs[i, o_i].set_axis_off()

plt.savefig('test2.png', dpi=300)
# plt.savefig(snakemake.output[2], dpi=300)
# plt.clf()

# frequency profile lfp

# f2_fig = plt.figure(figsize=(16, 9), dpi=300)
# f2_axs = f2_fig.subplots(len(ch), 4, gridspec_kw=dict(wspace=0.15, hspace=0.2))
# f2_fig.subplots_adjust(left=0.12, right=0.92, top=0.9, bottom=0.1)
#
# f_max = []
# f_min = []
#
# for j in range(len(lvp)):
#     pi = lvl.index(lvp[j])
#     for i in range(len(ch)):
#         # tvn = g_s[lvp[j]][7][ch[i]]
#         # tvc = g_s[lvp[j]][7][ch[i]]
#         tspn = g_n[lvp[j]][3][ch[i]]
#         tspc = g_s[lvp[j]][3][ch[i]]
#
#         f_max.append([np.max(tspc), np.max(tspn)])
#         f_min.append([np.min(tspn), np.min(tspc)])
#
# ymx = np.around(np.max(f_max) * 2/1000, decimals=2)
# ymn = np.around(np.min(f_min) * 1.3/1000, decimals=2)

# for j in lvp:
#     pi = lvl.index(j)
#     for i in range(len(ch)):
#         fa = f2_axs[i, pi]
#         tspn = g_n[j][3]/1000
#         tspc = g_s[j][3]/1000
#         tvn = g_n[j][7]/1000
#         tvc = g_s[j][7]/1000
#
#         p_dat = [tspc, tspn]
#         p_var = [tvc, tvn]
#         labelz = ['Control', 'NN']
#         ltype = ['-', '--']
#
#         for k in range(len(p_dat)):
#             fa.plot(octs, p_dat[k][ch[i]],
#                     label=labelz[k], lw=1,
#                     color=dcmap[k], linestyle=ltype[k]
#                     )
#             if 'all' in animal:
#                 pass
#             else:
#                 fa.fill_between(octs,
#                                 p_var[k][ch[i]][:, 0],
#                                 p_var[k][ch[i]][:, 1], alpha=.2
#                                 )
#
#         fa.set_xticks([-1.5, -1., -0.5, 0, 0.5, 1., 1.5])
#         fa.set_yticks([ymn, 0])
#         fa.set_ylim(top=ymx, bottom=ymn)
#         fa.set_xlim(left=-1.5, right=1.5)
#         fa.tick_params(axis='both', which='major', labelsize=8, length=0.9,
#                        width=0.5)
#         fa.grid(True, alpha=0.6, lw=0.5)
#
#         for p in [-0.5, 0.5]:
#             fa.axvline(x=p, color='#363b50', linestyle='-.', lw=0.7)
#
#         # significance
#         if impl[j][ch[i]] <= .05:
#             print(impl[j][ch[i]])
#             fa.scatter(0, tspn[ch[i]][12] - 0.05, marker=(5, 2),
#                        c='black',
#                        alpha=0.7)
#         else:
#             pass
#         # tick lable off
#         if i != len(ch) - 1:
#             fa.tick_params(labelbottom=False)
#
#         if j == lvp[0]:
#             fa.set_ylabel(depth[i] + r'$\mu$' + 'm' + '\n\n\n\n\n',
#                           fontsize=10)
#         else:
#             fa.tick_params(labelleft=False)
#
#         # legend box
#         if j == lvp[len(lvp) - 1] and i == 0:
#             fa.legend(prop={'size': 7.5},
#                       loc='lower right',
#                       bbox_to_anchor=(1, 0))
#         else:
#             pass
#
#         if i == 0:
#             fa.text(0.5, 1.5, f'{j} SPL dB',
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     # rotation='vertical',
#                     transform=fa.transAxes,
#                     fontdict={'weight': 'bold', 'size': 11}
#                     )
#         elif i == len(ch) - 1:
#             fa.text(0.5, -0.3, f'Octave',
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     # rotation='vertical',
#                     transform=fa.transAxes,
#                     fontdict={'weight': 'bold', 'size': 11}
#                     )
#
# f2_axs[1, 0].text(-0.4, -0.7, 'Depth\n\n\n\n\n\nAmplitude ' + r'(mV)',
#                   horizontalalignment='center',
#                   verticalalignment='center',
#                   rotation='vertical',
#                   transform=f2_axs[1, 0].transAxes,
#                   fontdict={'weight': 'bold', 'size': 11}
#                   )
#
# for j in lv_off:
#     o_i = lvl.index(j)
#     for i in range(len(ch)):
#         f2_axs[i, o_i].set_axis_off()
#
# plt.savefig('test3.png', dpi=300)
# # plt.savefig(snakemake.output[3], dpi=300)
# plt.clf()

# # frequency profile: difference between control and notch conditions
# f3_fig = plt.figure(figsize=(16, 9), dpi=300)
# f3_fig.subplots_adjust(left=0.12, right=0.92, top=0.9, bottom=0.1)
# f3_axs = f3_fig.subplots(len(ch), 4, gridspec_kw=dict(wspace=0.15, hspace=0.2))
#
# f_m = []
# fmd = 0
#
# for j in range(len(lvp)):
#     pi = lvl.index(lvp[j])
#     for i in range(len(ch)):
#         chs = ch[i]
#         tspn = g_n[lvp[j]][2][chs]
#         tspc = g_s[lvp[j]][2][chs]
#
#         per = (tspn - tspc) / tspc
#         per = [np.nan if infs == np.inf else infs for infs in per]
#
#         f_m.append(per)
#
# ymax = np.around(np.nanmax(f_m) * 1.3, decimals=1)
# ymin = np.around(np.nanmin(f_m) * 1.3, decimals=1)
#
# if ymin == 0:
#     ymin = -0.1
# if ymax == 0:
#     ymax = 0.1
#
# for j in lvp:
#     pi = lvl.index(j)
#     for i in range(len(ch)):
#         chs = ch[i]
#         fa = f3_axs[i, pi]
#         tspn = g_n[j][2][chs]
#         tspc = g_s[j][2][chs]
#
#         p_dat = (tspn - tspc) / tspc
#         p_dat = [np.nan if infs == np.inf else infs for infs in p_dat]
#
#         fa.plot(octs, p_dat,
#                 label='NN-Ctrl', lw=1,
#                 color=dcmap[0], linestyle='-'
#                 )
#
#         fa.set_xticks([-1.5, -1., -0.5, 0, 0.5, 1., 1.5])
#         fa.set_yticks([-1, 0, 1])
#         fa.set_yticks([-0.5, 0.5], minor=True)
#         fa.set_ylim(top=1.3, bottom=-1.0)
#         fa.set_xlim(left=-1.5, right=1.5)
#         fa.tick_params(axis='both', which='major', labelsize=8, length=3,
#                        width=1)
#         fa.grid(True, alpha=0.6, lw=0.5)
#
#         for p in [-0.5, 0.5]:
#             fa.axvline(x=p, color='#363b50', linestyle='-.', lw=0.7)
#
#         # tick lable off
#         if i != len(ch) - 1:
#             fa.tick_params(labelbottom=False)
#
#         if j == lvp[0]:
#             fa.set_ylabel(depth[i] + r'$\mu$' + 'm' + '\n\n\n\n\n',
#                           fontsize=10)
#         else:
#             fa.tick_params(labelleft=False)
#
#         # legend box
#         if j == lvp[len(lvp) - 1] and i == 0:
#             fa.legend(prop={'size': 7.5},
#                       loc='lower right',
#                       bbox_to_anchor=(1, 0))
#         else:
#             pass
#
#         if i == 0:
#             fa.text(0.5, 1.5, f'{j} SPL dB',
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     # rotation='vertical',
#                     transform=fa.transAxes,
#                     fontdict={'weight': 'bold', 'size': 11}
#                     )
#         elif i == len(ch) - 1:
#             fa.text(0.5, -0.3, f'Octave',
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     # rotation='vertical',
#                     transform=fa.transAxes,
#                     fontdict={'weight': 'bold', 'size': 11}
#                     )
#
# f3_axs[1, 0].text(-0.4, -0.7, 'Depth\n\n\n\n\n\n'
#                   + 'Firing rate change    ' + r'$(\frac{NN - Control}{'
#                                                r'Control})$',
#                   horizontalalignment='center',
#                   verticalalignment='center',
#                   rotation='vertical',
#                   transform=f3_axs[1, 0].transAxes,
#                   fontdict={'weight': 'bold', 'size': 11}
#                   )
#
# for j in lv_off:
#     o_i = lvl.index(j)
#     for i in range(len(ch)):
#         f3_axs[i, o_i].set_axis_off()
#
# plt.savefig('test4.png', dpi=300)
# # plt.savefig(snakemake.output[4], dpi=300)
# plt.clf()
