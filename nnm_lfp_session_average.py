import numpy as np
import h5py
import spike_tools as st
import pickle
import parameters as pm
import sys

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

s_conds = np.append(nnc.conds, nns.conds[0:2])
# aaa = animals.index('cr50_190724')
# print(aaa)
# dicts
g_l = {}

f_lfp = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
        '/data/dataset/lfp_fil/cr35_190403_lfp_fil.hdf5'

# f_lfp = snakemake.input[0]

# an = 0  # the number which represents animal name order
f_part = f_lfp.split('_')[4]  # file name
f_ani = f_part.split('/')[1]  # cr# of animal
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
        for i in range(len(acor[cond])):
            a_l = []
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
            print(f'{cond}:{lvl} is finished')
        print(f'{cond}_strf is finished')


#
with h5py.File(f_lfp, 'r') as l_5:
    # loading the tsp/lfp dataset

    # nnm_continuous

    for cond in nnc.conds:
        g_l[cond] = {}
        for i in range(len(acor[cond])):
            a_l = []
            lvl = lv[i]
            g_l[cond][lvl] = []
            print('hoy', acor[cond][i][an])
            if np.isnan(acor[cond][i][an]):
                print('hey')
                pass
            else:
                s_cord = datas[cond][i][an]
                print(cond, s_cord)
                lfp = l_5[cond][s_cord]
                lfps = st.lfp_strf(p_info, lfp, tw=200)
                llfp = np.array(lfps[0])

                g_l[cond][lvl].extend(llfp)
            print(f'{cond}:{lvl} is finished')
        print(f'{cond}_strf is finished')



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
            for i in range(len(acor[cond])):
                a_l = []
                lvl = lv[i]
                g_l[cond][pt][lvl] = []

                print('hoy', acor[cond][i][an])
                if np.isnan(acor[cond][i][an]):
                    print('hey')
                    pass

                else:
                    s_cord = datas[cond][i][an]

                    lfp = l_5[cond][s_cord]
                    lfps = st.lfp_strf(p_infop, lfp, tw=200)
                    llfp = np.array(lfps[0])
                    g_l[cond][pt][lvl].extend(llfp)
                print(f'{cond}, {pt}:{lvl} is finished')
            print(f'{cond}: {pt} is finished')
        print(f'{cond} is finished')


with open(snakemake.output[0], 'wb') as datas:
    pickle.dump(g_l, datas, protocol=pickle.HIGHEST_PROTOCOL)