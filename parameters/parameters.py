import numpy as np
import pickle
from matplotlib.colors import ListedColormap
import seaborn as sns

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
    animals = [
        'cr29_190228', 'cr30_190305', 'cr31_190312', 'cr33_190321',
        'cr50_190724', 'cr79_200211', 'cr90_201023', 'cr92_201023',
        'cr93_201110', 'cr94_201112', 'cr95_201117', 'cr96_201119',
        'cr97_201201', 'cr98_201203', 'cr101_201211', 'cr102_201214',
        'cr103_201215'
        ]

    p_info = np.load(param_path + 'time_index_multi_tones_500ms_180s.npy',
                     allow_pickle=True)

    octaves = np.array(
        [-1.5, -1.38, -1.25, -1.13, -1, -0.88, -0.75, -0.63, -0.5,
         -0.37, -0.25, -0.13, 0, 0.12, 0.25, 0.38, 0.5, 0.63, 0.75,
         0.87, 1, 1.12, 1.25, 1.37, 1.5])

    # best f for each animal
    bf = [
        6727, 9514, 8724, 5657, 7336, 8000,
        2594, 20749, 1834, 8724, 8000, 2378,
        8000, 5657, 16000, 16000, 6727
        ]
    # best f dict
    bf_dic = {
        'cr29': 6727, 'cr30': 9514, 'cr31': 8724, 'cr33': 5657,
        'cr50': 7336, 'cr79': 8000, 'cr90': 2594, 'cr92': 20749,
        'cr93': 1834, 'cr94': 8724, 'cr95': 8000, 'cr96': 2378,
        'cr97': 8000, 'cr98': 5657, 'cr101': 16000, 'cr102': 16000,
        'cr103': 6727
        }
    fps = 'mfpt'  # which frequency profile you are going to use: fpt/mfpt
    fpl = 'mfpl'  # which frequency profile you are going to use: fpl/mfpl
    # # activity time window for each animal
    # tw = [[12, 46], [10, 41], [10, 41], [10, 41], [10, 38], [10, 31], [10, 41]]


"""
continuous nn
"""


class Nnmcontfeat:
    # conditions
    conds = ['nnm_3_0']
    # conds = ['nnm_3_0', 'nnm_3_0_05', 'nnm_lef', 'nnm_hef']

    # condition dict
    lists = {
        'nnm_3_0': ['nnm_3_0', 'nnm_3_0_strf']
        # 'nnm_3_0_05': ['nnm_3_0_05', 'nnm_3_0_05_strf']
        # 'nnm_lef': ['nnm_lef', 'nnm_lef_strf'],
        # 'nnm_hef': ['nnm_hef', 'nnm_hef_strf']
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
                 nb=5, ch=None, ws=0.1, hs=0.2, ccmap=None, ratio=0.7,
                 dm=None):

        if ch is None:
            ch = np.array([4, 6, 8, 10, 12])
        if ccmap is None:
            ccmap = ListedColormap(sns.color_palette("mako", 256))
        elif ccmap == 'lfp':
            ccmap = ListedColormap(sns.color_palette("icefire", 256))
            # "vlag" "icefire"
        elif ccmap == 'csd':
            ccmap = ListedColormap(sns.color_palette("icefire_r", 256))

        self.mode = mode
        self.yl = yl
        self.cond = cond
        self.cond2 = cond2
        self.nb = nb
        self.ch = ch
        self.ws = ws
        self.hs = hs
        self.ccmap = ccmap
        self.ratio = ratio
        self.dm = dm

    def values(self):
        depths = ['0', '100', '200', '300', '400', '500', '600', '700',
                  '800', '900', '1000', '1100', '1200', '1300', '1400', '1500']

        l_color = ['#e6e6e6', '#4e69ab']
        cb = '#e6e6e6'
        cc = 0
        depth = []

        if self.dm is None:
            dcmap = sns.color_palette('colorblind').as_hex()
        else:
            dcmap = 'jet'

        for i in self.ch:
            depth.append(depths[i])

        if self.cond2 == 'tsp':
            cc = l_color[0]
        elif self.cond2 == 'lfp':
            cc = l_color[1]
        else:
            print('ni nun ong-i-gumung')

        return [self.mode, self.yl, self.cond, self.cond2, self.nb, self.ch,
                depth, dcmap, self.ws, self.hs, self.ratio, cc, cb, self.ccmap]


class Nn:
    with open('/home/bhc/codes/spike_snakemake/parameters/nn_index.pk',
              'rb') as a:
        nn_i = pickle.load(a)


def lfp_p(times):
    t_point = (np.array(times) * sr_lfp / 1000).astype(int)
    return t_point
