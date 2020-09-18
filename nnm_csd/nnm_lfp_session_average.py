import os,sys,inspect, pathlib
sys.path.insert(1, os.path.dirname(os.getcwd()))
import numpy as np
import h5py
from tools import spike_tools as st
import pickle
from parameters import parameters as pm
sys.path.pop(1)

'''
This is to average the lfp sessions of nnm conditions for csd. Possibly can 
be used for heatmap analysis
'''

'''
setup
'''
# classes
com = pm.Commonfeat  # common features
nnc = pm.Nnmcontfeat  # continuous notched noise features
nns = pm.Nnsifeat  # notch on-off features

# frequency list turned from octave to Hz
freq = com.list_freq

# any other variables are included in spike_tools
f_path = com.f_path

# animal/data folder name
animals = com.animals

# time index for strf
p_info = com.p_info
p_info1 = nns.p_info11
p_info3 = nns.p_info31

# data coordination of each animal
db = com.db

# for details, check data_list.ods
acor = db['animal']  # animal coordination
datas = db['datas']  # dB data set location in each animal dict
bf = db['bf']  # best frequency list of each animal
tw = db['tw']  # time window of activity in each animal
lv = db['lv']  # SPLdB list
dst = db['d_strf']  # strf which corresponds to that specific nnm file
p_type = ['nn', 'si']  # corresponds to dict keys for notched noise and #
# silence sessions.

s_conds = nnc.conds + nns.conds[0:2]
# aaa = animals.index('cr50_190724')
# print(aaa)
# dicts
g_l = {}
# average of all strf data in an animal
g_l['a_strf'] = {'70': [], '60': [], '50': [], '40': []}
sl2 = {'70': [], '60': [], '50': [], '40': []}
# f_lfp = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#         '/data/dataset/lfp_fil/cr35_190403_lfp_fil.hdf5'

f_lfp = snakemake.input[0]

# an = 0  # the number which represents animal name order
f_ani = f_lfp.split('/')[-1].split('_')[0]  # cr# of animal
an = 0  # an is the index of the animal used

for num_a, ani in enumerate(animals):
    if f_ani in ani:
        an = num_a
        break



print(f"{f_ani}: {an}")

with h5py.File(f_lfp, 'r') as l_5:
    # loading the tsp/lfp dataset

    # strf

    for cond in s_conds:
        g_l[cond + '_strf'] = {}
        for i in range(len(lv)):
            lvl = lv[i]
            g_l[cond + '_strf'][lvl] = []

            print('hoy', acor[cond][i][an])
            if np.isnan(acor[cond][i][an]):
                print('hey')
                pass
            else:
                s_cord = dst[cond][an]
                lfp = l_5[s_cord][i]
                lfps = st.lfp_strf(p_info, lfp, tw=200)
                llfp = np.array(lfps[0])
                g_l[cond + '_strf'][lvl].extend(llfp)
                print(np.shape(g_l[cond + '_strf'][lvl]))
            print(f'{cond}_strf:{lvl} is finished')
        print(f'{cond}_strf is finished')


with h5py.File(f_lfp, 'r') as l_5:
    # loading the tsp/lfp dataset

    # nnm_continuous

    for cond in nnc.conds:
        g_l[cond] = {}
        for i in range(len(lv)):
            lvl = lv[i]
            g_l[cond][lvl] = []
            print('hoy', acor[cond][i][an])
            if np.isnan(acor[cond][i][an]):
                print('hey')
                pass
            else:
                d_cord = datas[cond][i][an]
                print(cond, d_cord)
                lfp = l_5[cond][d_cord]
                lfps = st.lfp_strf(p_info, lfp, tw=200)
                llfp = np.array(lfps[0])
                sl2[lvl].append(llfp)
                g_l[cond][lvl].extend(llfp)
            print(f'{cond}:{lvl} is finished')
        print(f'{cond} is finished')

for lvl in lv:
    if not np.any(sl2[lvl]):
        continue
    else:
        g_l['a_strf'][lvl].extend(np.mean(sl2[lvl], axis=0))



with h5py.File(f_lfp, 'r') as l_5:
    # loading the tsp/lfp dataset
    # nnm_silence

    for cond in nns.conds:
        g_l[cond] = {}

        if cond == 'nnm_3_1' or cond == 'nnm_wn_3_1':
            p_infot = p_info3
        elif cond == 'nnm_1_1' or cond == 'nnm_wn_1_1':
            p_infot = p_info1
        else:
            'read the fucking script again you shithead'
            sys.exit()

        for pt in p_type:
            g_l[cond][pt] = {}
            p_infop = p_infot[pt]
            for i in range(len(lv)):
                lvl = lv[i]
                g_l[cond][pt][lvl] = []

                print('hoy', acor[cond][i][an])
                if np.isnan(acor[cond][i][an]):
                    print('hey')
                    pass

                else:
                    d_cord = datas[cond][i][an]

                    lfp = l_5[cond][d_cord]
                    lfps = st.lfp_strf(p_infop, lfp, tw=200)
                    llfp = np.array(lfps[0])
                    g_l[cond][pt][lvl].extend(llfp)
                print(f'{cond}, {pt}:{lvl} is finished')
            print(f'{cond}_{pt}: {lvl} is finished')
        print(f'{cond}_{pt} is finished')


# with open(snakemake.output[0], 'wb') as datas:
#     pickle.dump(g_l, datas, protocol=pickle.HIGHEST_PROTOCOL)
s_conds = s_conds + ['a']

with h5py.File(snakemake.output[0], 'w') as h:
    for cond in s_conds:
        d1 = h.create_group(f'{cond}_strf')
        for lvl in lv:
            d2 = d1.create_dataset(
                lvl, data=g_l[f'{cond}_strf'][lvl])

    for cond in nnc.conds:
        d1 = h.create_group(cond)
        for lvl in lv:
            d2 = d1.create_dataset(
                lvl, data=g_l[cond][lvl])

    for cond in nns.conds:
        d1 = h.create_group(cond)
        for pt in p_type:
            d2 = d1.create_group(pt)
            for lvl in lv:
                d3 = d2.create_dataset(
                    lvl, data=g_l[cond][pt][lvl])
