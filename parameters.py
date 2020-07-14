import numpy as np
import h5py
import spike_tools as st
import pickle
import notch_gap_params as p_tool

file_path = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/'
param_path = '/home/bhc/codes/spike_snakemake/parameters/'
sr_lfp = 1061.5


class Commonfeat:
    with open(param_path + 'animal_dB_coord.pk',
              'rb') as a_coord:
        db = pickle.load(a_coord)

    list_freq = np.load(param_path + 'list_freq.npy', allow_pickle=True)

    # any other variables are included in spike_tools
    f_path = file_path

    # animal/data folder name
    animals = ['cr29_190228', 'cr30_190305', 'cr31_190312', 'cr33_190321',
               'cr35_190403', 'cr50_190724', 'cr79_200211']

    p_info = np.load(param_path + 'time_index_multi_tones_500ms_180s.npy',
                     allow_pickle=True)

    octaves = np.array(
        [-1.5, -1.38, -1.25, -1.13, -1, -0.88, -0.75, -0.63, -0.5,
         -0.37, -0.25, -0.13, 0, 0.12, 0.25, 0.38, 0.5, 0.63, 0.75,
         0.87, 1, 1.12, 1.25, 1.37, 1.5])

    # best f for each animal
    bf = [6727, 9514, 8724, 5657, 2828, 7336, 8000]
    # best f dict
    bf_dic = {'cr29': 6727, 'cr30': 9514, 'cr31': 8724, 'cr33': 5657,
              'cr35': 2828, 'cr50': 7336, 'cr79': 8000}
    # activity time window for each animal
    tw = [[12, 46], [10, 41], [10, 41], [10, 41], [10, 38], [10, 31], [10, 41]]


"""
continuous nn
"""


class Nnmcontfeat:
    # conditions
    conds = ['nnm_3_0', 'nnm_3_0_05', 'nnm_lef', 'nnm_hef']

    # condition dict
    lists = {
        'nnm_3_0': ['nnm_3_0', 'nnm_3_0_strf'],
        'nnm_3_0_05': ['nnm_3_0_05', 'nnm_3_0_05_strf'],
        'nnm_lef': ['nnm_lef', 'nnm_lef_strf'],
        'nnm_hef': ['nnm_hef', 'nnm_hef_strf']
        }


"""
nn on-off
"""


class Nnsifeat:
    # timed index for multi-tone pip trigger

    with open(param_path + 'time_index_multi_tones_500ms_180s_nn_si.pk',
              'rb') as pip:
        p_infos = pickle.load(pip)

    p_info31 = {'nn': p_infos['t_index_31_nn'], 'si': p_infos['t_index_31_si']}

    p_info11 = {'nn': p_infos['t_index_11_nn'], 'si': p_infos['t_index_11_si']}

    # conditions
    conds = ['nnm_3_1', 'nnm_1_1', 'nnm_wn_3_1', 'nnm_wn_1_1']

    # condition dicts
    lists = {
        'nnm_3_1': ['nnm_3_1', 'nnm_3_1_strf', 'nnm_wn_3_1'],
        'nnm_1_1': ['nnm_1_1', 'nnm_1_1_strf', 'nnm_wn_1_1'],
        }


class Datas:
    try:
        with open(file_path + 'average_tsp.pk', 'rb') as a:
            nnm_t = pickle.load(a)

        with open(file_path + 'average_lfp.pk', 'rb') as a:
            nnm_l = pickle.load(a)

        with open(file_path + 'average_freq.pk', 'rb') as a:
            nnm_f = pickle.load(a)

    except FileNotFoundError:
        pass


class Setup:
    def __init__(self, mode=0, yl=0.1, cond=None, cond2=None,
                 nb=5, ch=None, ws=0.1, hs=0.15, ccmap="jet"):
        if ch is None:
            ch = np.array([1, 5, 8, 10, 12])

        self.mode = mode
        self.yl = yl
        self.cond = cond
        self.cond2 = cond2
        self.nb = nb
        self.ch = ch
        self.ws = ws
        self.hs = hs
        self.ccmap = ccmap

    def values(self):
        depths = ['0', '100', '200', '300', '400', '500', '600', '700',
                  '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']

        l_color = ['#e6e6e6', '#4e69ab']
        cb = '#e6e6e6'
        cc = 0
        depth = []

        for i in self.ch:
            depth.append(depths[i])

        if self.cond2 == 'tsp':
            cc = l_color[0]
        elif self.cond2 == 'lfp':
            cc = l_color[1]
        else:
            print('ni nun ong-i-gumung')

        return [self.mode, self.yl, self.cond, self.cond2, self.nb, self.ch,
                depth, self.ws, self.hs, self.ccmap, cc, cb]


class Nn:
    with open('/home/bhc/codes/spike_snakemake/parameters/nn_index.pk',
              'rb') as a:
        nn_i = pickle.load(a)


def lfp_p(times):
    t_point = (np.array(times) * sr_lfp / 1000).astype(int)
    return t_point
