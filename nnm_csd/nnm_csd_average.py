import os
import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))
import numpy as np
import pickle
import h5py
from parameters import parameters as pm
import quantities as pq
import neo.core
from tools import icsd
sys.path.pop(1)

file_input = snakemake.input
# file_input = ["/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr29_190228_lfp_avg_nnm.hdf5"
#               , "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr30_190305_lfp_avg_nnm.hdf5", "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr31_190312_lfp_avg_nnm.hdf5", "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr33_190321_lfp_avg_nnm.hdf5", "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr50_190724_lfp_avg_nnm.hdf5", "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr79_200211_lfp_avg_nnm.hdf5",
# "/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021"
#               "/data/dataset/nnm_lfp_avg/cr90_201023_lfp_avg_nnm.hdf5"
#               ]

# classes
com = pm.Commonfeat  # common features
nnc = pm.Nnmcontfeat  # continuous notched noise features
nns = pm.Nnsifeat  # notch on-off features

# frequency list turned from octave to Hz
freq = com.list_freq

# any other variables are included in spike_tools
f_path = com.f_path

# animal/data folder name
animals = []
for i in range(len(file_input)):
    name = file_input[i].split('/')[-1].split('_')[0]
    animals = np.append(animals, name)


sr_lfp = 1061.5 * pq.Hz  # for convert time into data coord in the data file
# data coordination of each animal
db = com.db
lv = db['lv']  # SPLdB list
# for details, check data_list.ods
acor = db['animal']  # animal coordination

bfs = com.bf_dic  # best frequency dictionary
b_list = com.list_freq  # list of the 1/8 octave frequency to find index
# csd parameters
coord = np.arange(0, 1501, 100) * pq.micrometer
sigma = 1 / (100 * pq.micrometer)
# silence sessions.
s_conds = np.append(nnc.conds, nns.conds[0:2])

'''
strf
'''
g_l = {}
g_c = {}

for cond in s_conds:
    cond_s = f'{cond}_strf'
    g_l[cond_s] = {}
    g_c[cond_s] = {}
    # g_l[f'{cond}_astrf'] = {}
    # g_c[f'{cond}_astrf'] = {}
    for i in range(len(lv)):
        a_l = []
        a_c = []
        # a_la = []
        # a_ca = []
        lvl = lv[i]
        g_l[cond_s][lvl] = {}
        g_c[cond_s][lvl] = {}
        # g_l[f'{cond}_astrf'][lvl] = {}
        # g_c[f'{cond}_astrf'][lvl] = {}
        for j in acor[cond][i]:
            a_l2 = []
            # a_l3 = []
            if np.isnan(j):
                continue
            else:
                print(j, animals)
                animal = animals[j]
                bf = bfs[animal]
                if cond == 'nnm_hef':
                    cf = np.round(bf * 2 ** (-0.5))
                    if cf == 9513:
                        cf = 9514
                    elif cf == 3999:
                        cf = 4000
                    else:
                        pass
                    b_ind = np.where(b_list == cf)[0][0]

                elif cond == 'nnm_lef':
                    cf = np.round(bf * 2 ** 0.5)
                    if cf == 9513:
                        cf = 9514
                    elif cf == 3999:
                        cf = 4000
                    else:
                        pass
                    b_ind = np.where(b_list == cf)[0][0]
                else:
                    b_ind = np.where(b_list == bf)[0][0]

                with h5py.File(file_input[j], 'r') as dat:
                    datas = dat
                    la = np.array(datas[cond_s][lvl])
                    # la2 = np.array(datas['a_strf'][lvl])
                    print(animal, cond_s)
                    for p in range(len(la)):
                        a_l2.append(la[p][b_ind])
                        # a_l3.append(la2[p][b_ind])
                    lu = a_l2 * pq.microvolt
                    # lu2 = a_l3 * pq.mV
                    lfp_data1 = neo.core.AnalogSignal(lu.T,
                                                      sampling_rate=sr_lfp)
                    # lfp_data2 = neo.core.AnalogSignal(lu2.T,
                    #                                   sampling_rate=sr_lfp)
                    std_input_nnm = {
                        'lfp': lfp_data1,
                        'coord_electrode': coord,
                        'sigma': sigma,
                        'f_type': 'gaussian',
                        'f_order': (3, 1)
                        }

                    # std_input_nnm2 = {
                    #     'lfp': lfp_data2,
                    #     'coord_electrode': coord,
                    #     'sigma': sigma,
                    #     'f_type': 'gaussian',
                    #     'f_order': (3, 1)
                    #     }

                    std1 = icsd.estimate_csd(**std_input_nnm)
                    fs_csd = np.transpose(std1[1] / std1[1].units) * \
                             std1[1].units
                    fs_csd.units = pq.mV / (pq.mm ** 2)

                    # std2 = icsd.estimate_csd(**std_input_nnm2)
                    # fs_csd2 = np.transpose(std2[1] / std2[1].units) * \
                    #          std2[1].units
                    # fs_csd2.units = pq.mV / (pq.mm ** 2)

                    a_l.append(a_l2)
                    a_c.append(fs_csd)
                    # a_la.append(a_l2)
                    # a_ca.append(fs_csd)

        g_l[cond_s][lvl] = np.mean(a_l, axis=0)
        g_c[cond_s][lvl] = np.mean(a_c, axis=0)
        # g_l[f'{cond}_astrf'][lvl] = np.mean(a_la, axis=0)
        # g_c[f'{cond}_astrf'][lvl] = np.mean(a_ca, axis=0)
        print(f'strf {cond}: {lvl} finished')

# for cond in ['nnm_3_0', 'nnm_3_0_05']:
for cond in ['nnm_3_0']:
    g_l[cond] = {}
    g_c[cond] = {}
    for i in range(len(lv)):
        a_l = []
        a_c = []
        lvl = lv[i]
        g_l[cond][lvl] = {}
        g_c[cond][lvl] = {}
        for j in acor[cond][i]:
            a_l2 = []
            if np.isnan(j):
                continue
            else:
                animal = animals[j]
                bf = bfs[animal]
                print(bf)
                b_ind = np.where(b_list == bf)[0][0]

                with h5py.File(file_input[j], 'r') as dat:
                    datas = dat
                    la = np.array(datas[cond][lvl])
                    for p in range(len(la)):
                        a_l2.append(la[p][b_ind])
                    lu = a_l2 * pq.microvolt
                    lfp_data1 = neo.core.AnalogSignal(lu.T,
                                                      sampling_rate=sr_lfp)
                    std_input_nnm = {
                        'lfp': lfp_data1,
                        'coord_electrode': coord,
                        'sigma': sigma,
                        'f_type': 'gaussian',
                        'f_order': (3, 1)
                        }

                    std1 = icsd.estimate_csd(**std_input_nnm)
                    fs_csd = np.transpose(std1[1] / std1[1].units) * std1[
                        1].units
                    fs_csd.units = pq.mV / (pq.mm ** 2)
                    a_l.append(a_l2)
                    a_c.append(fs_csd)

        g_l[cond][lvl] = np.mean(a_l, axis=0)
        g_c[cond][lvl] = np.mean(a_c, axis=0)
        print(f'{cond}: {lvl} finished')

# for cond in ['nnm_hef', 'nnm_lef']:
#     g_l[cond] = {}
#     g_c[cond] = {}
#     for i in range(len(lv)):
#         a_l = []
#         a_c = []
#         lvl = lv[i]
#         g_l[cond][lvl] = {}
#         g_c[cond][lvl] = {}
#         for j in acor[cond][i]:
#             a_l2 = []
#             if np.isnan(j):
#                 continue
#             else:
#                 animal = animals[j]
#                 bf = bfs[animal]
#                 cf = 0
#                 if cond == 'nnm_hef':
#                     cf = np.round(bf * 2 ** (-0.5))
#                 elif cond == 'nnm_lef':
#                     cf = np.round(bf * 2 ** 0.5)
#                 else:
#                     print(f'there is no cf for this {cond}, {lvl}')
#                     sys.exit()
#                 print(cf)
#                 if cf == 9513:
#                     cf = 9514
#                 elif cf == 3999:
#                     cf = 4000
#                 else:
#                     pass
#                 b_ind = np.where(b_list == cf)[0][0]
#
#                 with h5py.File(file_input[j], 'r') as dat:
#                     datas = dat
#                     la = np.array(datas[cond][lvl])
#                     for p in range(len(la)):
#                         a_l2.append(la[p][b_ind])
#                     lu = a_l2 * pq.mV
#                     lfp_data1 = neo.core.AnalogSignal(lu.T,
#                                                       sampling_rate=sr_lfp)
#                     std_input_nnm = {
#                         'lfp': lfp_data1,
#                         'coord_electrode': coord,
#                         'sigma': sigma,
#                         'f_type': 'gaussian',
#                         'f_order': (3, 1)
#                         }
#
#                     std1 = icsd.estimate_csd(**std_input_nnm)
#                     fs_csd = np.transpose(std1[1] / std1[1].units) * std1[
#                         1].units
#                     fs_csd.units = pq.mV / (pq.mm ** 2)
#                     a_l.append(a_l2)
#                     a_c.append(fs_csd)
#
#         g_l[cond][lvl] = np.mean(a_l, axis=0)
#         g_c[cond][lvl] = np.mean(a_c, axis=0)
#         print(f'{cond} {lvl} finished')

for cond in ['nnm_3_1', 'nnm_1_1', 'nnm_wn_3_1', 'nnm_wn_1_1']:
    g_l[cond] = {}
    g_c[cond] = {}
    for i in range(len(lv)):
        a_ln = []
        a_cn = []
        a_ls = []
        a_cs = []
        lvl = lv[i]
        g_l[cond][lvl] = {}
        g_c[cond][lvl] = {}
        for j in acor[cond][i]:
            a_ln2 = []
            a_ls2 = []
            if np.isnan(j):
                continue
            else:
                animal = animals[j]
                bf = bfs[animal]
                print(bf)
                b_ind = np.where(b_list == bf)[0][0]

                with h5py.File(file_input[j], 'r') as dat:
                    datas = dat
                    lan = np.array(datas[cond]['nn'][lvl])
                    las = np.array(datas[cond]['si'][lvl])
                    for p in range(len(la)):
                        a_ln2.append(lan[p][b_ind])
                        a_ls2.append(las[p][b_ind])

                    lun = a_ln2 * pq.microvolt
                    lus = a_ls2 * pq.microvolt
                    lfp_data1 = neo.core.AnalogSignal(lun.T,
                                                      sampling_rate=sr_lfp)
                    lfp_data2 = neo.core.AnalogSignal(lus.T,
                                                      sampling_rate=sr_lfp)
                    std_input_nn = {
                        'lfp': lfp_data1,
                        'coord_electrode': coord,
                        'sigma': sigma,
                        'f_type': 'gaussian',
                        'f_order': (3, 1)
                        }

                    std_input_si = {
                        'lfp': lfp_data2,
                        'coord_electrode': coord,
                        'sigma': sigma,
                        'f_type': 'gaussian',
                        'f_order': (3, 1)
                        }
                    std1 = icsd.estimate_csd(**std_input_nn)
                    std2 = icsd.estimate_csd(**std_input_si)
                    f1s_csd = np.transpose(std1[1] / std1[1].units) * std1[
                        1].units
                    f1s_csd.units = pq.mV / (pq.mm ** 2)
                    f2s_csd = np.transpose(std2[1] / std2[1].units) * std2[
                        1].units
                    f2s_csd.units = pq.mV / (pq.mm ** 2)
                    a_ln.append(a_ln2)
                    a_cn.append(f1s_csd)
                    a_ls.append(a_ls2)
                    a_cs.append(f2s_csd)

        g_l[cond][lvl] = [np.mean(a_ln, axis=0), np.mean(a_ls, axis=0)]
        g_c[cond][lvl] = [np.mean(a_cn, axis=0), np.mean(a_cs, axis=0)]
        print(f'{cond} {lvl} finished')

g_t = {'lfp': g_l, 'csd': g_c}
with open(snakemake.output[0], 'wb') as sav:
    pickle.dump(g_t, sav, protocol=pickle.HIGHEST_PROTOCOL)

with h5py.File(snakemake.output[0], 'w') as h:
    for t in list(g_t.keys()):
        d1 = h.create_group(t)
        for cond in s_conds:
            d2 = d1.create_group(f'{cond}_strf')
            for lvl in lv:
                d3 = d2.create_dataset(
                    lvl, data=g_t[t][f'{cond}_strf'][lvl])

        for cond in nnc.conds:
            d2 = d1.create_group(cond)
            for lvl in lv:
                d3 = d2.create_dataset(
                    lvl, data=g_t[t][cond][lvl])

        for cond in nns.conds:
            d2 = d1.create_group(cond)
            for lvl in lv:
                d3 = d2.create_dataset(
                    lvl, data=g_t[t][cond][lvl])
