
######## snakemake preamble start (automatically inserted, do not edit) ########
import sys; sys.path.extend(['/home/bhc/anaconda3/lib/python3.8/site-packages', '/home/bhc/codes/spike_snakemake/nnm_spike_lfp']); import pickle; snakemake = pickle.loads(b'\x80\x04\x95\x1f\x05\x00\x00\x00\x00\x00\x00\x8c\x10snakemake.script\x94\x8c\tSnakemake\x94\x93\x94)\x81\x94}\x94(\x8c\x05input\x94\x8c\x0csnakemake.io\x94\x8c\nInputFiles\x94\x93\x94)\x81\x94(\x8ci/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/dataset/lfp_fil/cr50_190724_lfp_fil.hdf5\x94\x8ca/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/dataset/tsp/cr50_190724_tsp.hdf5\x94e}\x94(\x8c\x06_names\x94}\x94(\x8c\x03lfp\x94K\x00N\x86\x94\x8c\x03tsp\x94K\x01N\x86\x94u\x8c\x12_allowed_overrides\x94]\x94(\x8c\x05index\x94\x8c\x04sort\x94eh\x15\x8c\tfunctools\x94\x8c\x07partial\x94\x93\x94h\x06\x8c\x19Namedlist._used_attribute\x94\x93\x94\x85\x94R\x94(h\x1b)}\x94\x8c\x05_name\x94h\x15sNt\x94bh\x16h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x16sNt\x94bh\x0fh\nh\x11h\x0bub\x8c\x06output\x94h\x06\x8c\x0bOutputFiles\x94\x93\x94)\x81\x94\x8cy/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data/dataset/nnm_spk_lfp_avg/cr50_190724_spk_lfp_avg_nnm.hdf5\x94a}\x94(h\r}\x94h\x13]\x94(h\x15h\x16eh\x15h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x15sNt\x94bh\x16h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x16sNt\x94bub\x8c\x06params\x94h\x06\x8c\x06Params\x94\x93\x94)\x81\x94}\x94(h\r}\x94h\x13]\x94(h\x15h\x16eh\x15h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x15sNt\x94bh\x16h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x16sNt\x94bub\x8c\twildcards\x94h\x06\x8c\tWildcards\x94\x93\x94)\x81\x94\x8c\x0bcr50_190724\x94a}\x94(h\r}\x94\x8c\x04name\x94K\x00N\x86\x94sh\x13]\x94(h\x15h\x16eh\x15h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x15sNt\x94bh\x16h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x16sNt\x94b\x8c\x04name\x94hHub\x8c\x07threads\x94K\x01\x8c\tresources\x94h\x06\x8c\tResources\x94\x93\x94)\x81\x94(K\x01K\x01e}\x94(h\r}\x94(\x8c\x06_cores\x94K\x00N\x86\x94\x8c\x06_nodes\x94K\x01N\x86\x94uh\x13]\x94(h\x15h\x16eh\x15h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x15sNt\x94bh\x16h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x16sNt\x94bh^K\x01h`K\x01ub\x8c\x03log\x94h\x06\x8c\x03Log\x94\x93\x94)\x81\x94}\x94(h\r}\x94h\x13]\x94(h\x15h\x16eh\x15h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x15sNt\x94bh\x16h\x19h\x1b\x85\x94R\x94(h\x1b)}\x94h\x1fh\x16sNt\x94bub\x8c\x06config\x94}\x94\x8c\x04rule\x94\x8c\x17spk_lfp_session_average\x94\x8c\x0fbench_iteration\x94N\x8c\tscriptdir\x94\x8c-/home/bhc/codes/spike_snakemake/nnm_spike_lfp\x94ub.'); from snakemake.logging import logger; logger.printshellcmds = False; __real_file__ = __file__; __file__ = '/home/bhc/codes/spike_snakemake/nnm_spike_lfp/spk_lfp_session_average.py';
######## snakemake preamble end #########
import os
import sys
sys.path.insert(1, os.path.dirname(os.getcwd()))
import numpy as np
import h5py
import pickle
from tools import spike_tools as st
from parameters import notch_gap_params as p_tool
from parameters import parameters as pm
import sys
sys.path.pop(1)

'''
This is creating average of all the animal data. 
std and confidence interval (0.95) is between time window used for generate 
frequency profile
recording session.
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
p_type = ['nn', 'si']  # corresponds to dict keys for notched noise and

# silence sessions.

s_conds = nnc.conds + nns.conds[0:2]

# dicts
g_t = {}  # tsp, tsp session average dict
g_l = {}  # lfp, lfp session average dict
g_f = {}  # fpt, spike frequency profile dict
g_mf = {}  # fpt, spike frequency profile dict with max value
g_s = {}  # ssd, spike frequency profile standard deviation dict
g_cf = {}  # scf, spike frequency profile confidence interval(95%) dict
g_lf = {}  # fpl, lfp frequency profile dict
g_mlf = {}  # fpl, lfp frequency profile dict with min value
g_mlb = {}  # baseline for lfp frequency profile
g_ls = {}  # lsd, lfp frequency profile standard deviation dict
g_lcf = {}  # lcf, lfp frequency profile confidence interval(95%) of mean dict
list_o = {}  # list of octave

# data to load

# f_lfp = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#         '/data/dataset/lfp_fil/cr101_201211_lfp_fil.hdf5'
# f_tsp = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
#         '/dataset/tsp/cr101_201211_tsp.hdf5'

f_lfp = snakemake.input.lfp
f_tsp = snakemake.input.tsp

f_ani = f_lfp.split('/')[-1].split('_')[0]  # cr# of animal

an = 0  # the index of the animal used in this script

for num_a, ani in enumerate(animals):
    if f_ani in ani:
        an = num_a
        break

print(f"{f_ani}: {an}")

if f_ani in ['cr92', 'cr93']:
    with h5py.File(snakemake.output[0], 'w') as h:
        d2 = h.create_dataset('na', data=[])
    sys.exit()


'''
strf
'''

for cond in s_conds:
    g_t[cond + '_strf'] = {}
    g_l[cond + '_strf'] = {}
    g_f[cond + '_strf'] = {}
    g_mf[cond + '_strf'] = {}
    g_s[cond + '_strf'] = {}
    g_cf[cond + '_strf'] = {}
    g_lf[cond + '_strf'] = {}
    g_mlf[cond + '_strf'] = {}
    g_mlb[cond + '_strf'] = {}
    g_ls[cond + '_strf'] = {}
    g_lcf[cond + '_strf'] = {}
    list_o[cond + '_strf'] = {}
    for i in range(len(lv)):
        lvl = lv[i]
        g_t[cond + '_strf'][lvl] = []
        g_l[cond + '_strf'][lvl] = []
        g_f[cond + '_strf'][lvl] = []
        g_mf[cond + '_strf'][lvl] = []
        g_s[cond + '_strf'][lvl] = []
        g_cf[cond + '_strf'][lvl] = []
        g_lf[cond + '_strf'][lvl] = []
        g_mlf[cond + '_strf'][lvl] = []
        g_mlb[cond + '_strf'][lvl] = []
        g_ls[cond + '_strf'][lvl] = []
        g_lcf[cond + '_strf'][lvl] = []
        list_o[cond + '_strf'][lvl] = []

        if np.isnan(acor[cond][i][an]):
            print('hey')
            continue

        else:
            s_cord = dst[cond][an]
            bfs = bf[an]
            oct_r = p_tool.o_cut(bfs, option=cond)

            # octave range calculation
            freq_o = np.around(np.log2(freq / bfs), decimals=2)
            list_o[cond + '_strf'][lvl].extend(freq_o[oct_r[0]:oct_r[1] + 1])

            # select spk activation time window
            t_w = tw[an]

            # loading the tsp/lfp dataset
            with h5py.File(f_tsp, 'r') as h_5:
                time_snip = h_5[s_cord][i]
                mua = st.index_strf(time_snip)
                tsps = st.tsp_strf(p_info, mua, twi=t_w[0], twf=t_w[1])
                ts = np.array(tsps[0])  # temporal spk data
                at = np.array(tsps[1])  # frequency profile
                mt = np.array(tsps[4])  # frequency profile with max value
                std = np.array(tsps[2])  # std of freqency profile
                cof = np.array(tsps[3])  # conf of freqency profile

                g_t[f'{cond}_strf'][lvl].extend(
                    ts[:, oct_r[0]:oct_r[1] + 1, :]
                    )
                g_f[f'{cond}_strf'][lvl].extend(
                    at[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_mf[f'{cond}_strf'][lvl].extend(
                    mt[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_s[f'{cond}_strf'][lvl].extend(
                    std[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_cf[f'{cond}_strf'][lvl].extend(
                    cof[:, oct_r[0]:oct_r[1] + 1]
                    )

                print(f_ani, cond, lvl)
            with h5py.File(f_lfp, 'r') as l_5:
                lfp = l_5[s_cord][i]
                lfps = st.lfp_strf(p_info, lfp, tw=100, twi=t_w[0], twf=t_w[1])
                lf = np.array(lfps[0])  # temporal lfp data
                al = np.array(lfps[1])  # frequency profile
                ml = np.array(lfps[4])  # frequency profile with min val
                sl = np.array(lfps[2])  # std of freqency profile
                cofl = np.array(lfps[3])  # conf of freqency profile
                base = np.array(lfps[5])

                g_l[f'{cond}_strf'][lvl].extend(
                    lf[:, oct_r[0]:oct_r[1] + 1, :]
                    )
                g_lf[f'{cond}_strf'][lvl].extend(
                    al[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_mlf[f'{cond}_strf'][lvl].extend(
                    ml[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_mlb[f'{cond}_strf'][lvl].extend(
                    base[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_ls[f'{cond}_strf'][lvl].extend(
                    sl[:, oct_r[0]:oct_r[1] + 1]
                    )
                g_lcf[f'{cond}_strf'][lvl].extend(
                    cofl[:, oct_r[0]:oct_r[1] + 1]
                    )

        print(f'{cond}_strf:{lvl} is finished')
    print(f'{cond}_strf is finished')


'''
nnm continuous
'''

for cond in nnc.conds:
    g_t[cond] = {}
    g_l[cond] = {}
    g_f[cond] = {}
    g_mf[cond] = {}
    g_lf[cond] = {}
    g_mlf[cond] = {}
    g_mlb[cond] = {}
    g_s[cond] = {}
    g_ls[cond] = {}
    g_cf[cond] = {}
    g_lcf[cond] = {}
    list_o[cond] = {}
    for i in range(len(lv)):
        lvl = lv[i]
        g_t[cond][lvl] = []
        g_l[cond][lvl] = []
        g_f[cond][lvl] = []
        g_mf[cond][lvl] = []
        g_lf[cond][lvl] = []
        g_mlf[cond][lvl] = []
        g_mlb[cond][lvl] = []
        g_s[cond][lvl] = []
        g_ls[cond][lvl] = []
        g_cf[cond][lvl] = []
        g_lcf[cond][lvl] = []
        list_o[cond][lvl] = []

        if np.isnan(acor[cond][i][an]):
            print('hey')
            continue

        else:
            d_cord = datas[cond][i][an]
            bfs = bf[an]
            oct_r = p_tool.o_cut(bfs, option=cond)

            # octave range calculation
            freq_o = np.around(np.log2(freq / bfs), decimals=2)
            list_o[cond][lvl].extend(freq_o[oct_r[0]:oct_r[1] + 1])

            # select spk activation time window
            t_w = tw[an]

            # loading the tsp/lfp dataset
            with h5py.File(f_tsp, 'r') as h_5:
                time_snip = h_5[cond][d_cord]
                mua = st.index_strf(time_snip)
                tsps = st.tsp_strf(p_info, mua, twi=t_w[0], twf=t_w[1])
                ts = np.array(tsps[0])  # temporal spk data
                at = np.array(tsps[1])  # frequency profile
                mt = np.array(tsps[4])  # frequency profile with max value
                std = np.array(tsps[2])  # std of freqency profile
                cof = np.array(tsps[3])  # cof of freqency profile


                g_t[cond][lvl].extend(ts[:, oct_r[0]:oct_r[1] + 1, :])
                g_f[cond][lvl].extend(at[:, oct_r[0]:oct_r[1] + 1])
                g_mf[cond][lvl].extend(mt[:, oct_r[0]:oct_r[1] + 1])
                g_s[cond][lvl].extend(std[:, oct_r[0]:oct_r[1] + 1])
                g_cf[cond][lvl].extend(cof[:, oct_r[0]:oct_r[1] + 1])

            with h5py.File(f_lfp, 'r') as l_5:
                lfp = l_5[cond][d_cord]
                lfps = st.lfp_strf(p_info, lfp, tw=100, twi=t_w[0], twf=t_w[1])
                lf = np.array(lfps[0])  # temporal spk data
                al = np.array(lfps[1])  # frequency profile
                ml = np.array(lfps[4])  # frequency profile with min val
                sl = np.array(lfps[2])  # std of freqency profile
                cofl = np.array(lfps[3])  # cof of freqency profile
                base = np.array(lfps[5])

                g_l[cond][lvl].extend(lf[:, oct_r[0]:oct_r[1] + 1, :])
                g_lf[cond][lvl].extend(al[:, oct_r[0]:oct_r[1] + 1])
                g_mlf[cond][lvl].extend(ml[:, oct_r[0]:oct_r[1] + 1])
                g_mlb[cond][lvl].extend(base[:, oct_r[0]:oct_r[1] + 1])
                g_ls[cond][lvl].extend(sl[:, oct_r[0]:oct_r[1] + 1])
                g_lcf[cond][lvl].extend(cofl[:, oct_r[0]:oct_r[1] + 1])

        print(f'{cond}:{lvl} is finished')
    print(f'{cond} is finished')

'''
nnm on-off
'''

for cond in nns.conds:
    g_t[cond] = {}
    g_l[cond] = {}
    g_f[cond] = {}
    g_mf[cond] = {}
    g_lf[cond] = {}
    g_mlf[cond] = {}
    g_mlb[cond] = {}
    g_s[cond] = {}
    g_ls[cond] = {}
    g_cf[cond] = {}
    g_lcf[cond] = {}
    list_o[cond] = {}

    if cond == 'nnm_3_1' or cond == 'nnm_wn_3_1':
        p_infot = p_info3
    elif cond == 'nnm_1_1' or cond == 'nnm_wn_1_1':
        p_infot = p_info1
    else:
        'read the fucking script again you shithead'
        sys.exit()
    for pt in p_type:
        g_t[cond][pt] = {}
        g_l[cond][pt] = {}
        g_f[cond][pt] = {}
        g_mf[cond][pt] = {}
        g_lf[cond][pt] = {}
        g_mlf[cond][pt] = {}
        g_mlb[cond][pt] = {}
        g_s[cond][pt] = {}
        g_ls[cond][pt] = {}
        g_cf[cond][pt] = {}
        g_lcf[cond][pt] = {}
        list_o[cond][pt] = {}
        p_infop = p_infot[pt]

        for i in range(len(lv)):
            lvl = lv[i]
            g_t[cond][pt][lvl] = []
            g_l[cond][pt][lvl] = []
            g_f[cond][pt][lvl] = []
            g_mf[cond][pt][lvl] = []
            g_lf[cond][pt][lvl] = []
            g_mlf[cond][pt][lvl] = []
            g_mlb[cond][pt][lvl] = []
            g_s[cond][pt][lvl] = []
            g_ls[cond][pt][lvl] = []
            g_cf[cond][pt][lvl] = []
            g_lcf[cond][pt][lvl] = []
            list_o[cond][pt][lvl] = []
            if np.isnan(acor[cond][i][an]):
                print('hey')
                continue

            else:
                d_cord = datas[cond][i][an]
                bfs = bf[an]
                oct_r = p_tool.o_cut(bfs, option=cond)

                # octave range calculation
                freq_o = np.around(np.log2(freq / bfs), decimals=2)
                list_o[cond][pt][lvl].extend(freq_o[oct_r[0]:oct_r[1] + 1])

                # select spk activation time window
                t_w = tw[an]

                # loading the tsp/lfp dataset
                with h5py.File(f_tsp, 'r') as h_5:
                    time_snip = h_5[cond][d_cord]
                    mua = st.index_strf(time_snip)
                    tsps = st.tsp_strf(p_infop, mua, twi=t_w[0], twf=t_w[1])
                    ts = np.array(tsps[0])  # temporal spk data
                    at = np.array(tsps[1])  # frequency profile
                    mt = np.array(tsps[4])  # frequency profile with max value
                    std = np.array(tsps[2])  # std of freqency profile
                    cof = np.array(tsps[3])  # cof of freqency profile

                    g_t[cond][pt][lvl].extend(ts[:, oct_r[0]:oct_r[1] + 1, :])
                    g_f[cond][pt][lvl].extend(at[:, oct_r[0]:oct_r[1] + 1])
                    g_mf[cond][pt][lvl].extend(mt[:, oct_r[0]:oct_r[1] + 1])
                    g_s[cond][pt][lvl].extend(std[:, oct_r[0]:oct_r[1] + 1])
                    g_cf[cond][pt][lvl].extend(cof[:, oct_r[0]:oct_r[1] + 1])

                with h5py.File(f_lfp, 'r') as l_5:
                    lfp = l_5[cond][d_cord]
                    lfps = st.lfp_strf(p_infop, lfp, tw=100, twi=t_w[0],
                                       twf=t_w[1])
                    lf = np.array(lfps[0])
                    al = np.array(lfps[1])
                    ml = np.array(lfps[4])  # frequency profile with min val
                    sl = np.array(lfps[2])
                    cofl = np.array(lfps[3])
                    base = np.array(lfps[5])
                    print(np.shape(cofl), 'check')
                    g_l[cond][pt][lvl].extend(lf[:, oct_r[0]:oct_r[1] + 1, :])
                    g_lf[cond][pt][lvl].extend(al[:, oct_r[0]:oct_r[1] + 1])
                    g_mlf[cond][pt][lvl].extend(ml[:, oct_r[0]:oct_r[1] + 1])
                    g_mlb[cond][pt][lvl].extend(base[:, oct_r[0]:oct_r[1] + 1])
                    g_ls[cond][pt][lvl].extend(sl[:, oct_r[0]:oct_r[1] + 1])
                    g_lcf[cond][pt][lvl].extend(cofl[:, oct_r[0]:oct_r[1] + 1])

                print(np.shape(g_l[cond][pt][lvl]))
            print(f'{cond}_{pt}:{lvl} is finished')
        print(f'{cond}_{pt} is finished')

groups = {'tsp': g_t, 'fpt': g_f, 'lfp': g_l, 'fpl': g_lf, 'ssd': g_s,
          'lsd': g_ls, 'scf': g_cf, 'lcf': g_lcf, 'mfpt': g_mf,
          'mfpl': g_mlf, 'base': g_mlb
          }
#
with h5py.File(snakemake.output[0], 'w') as h:
    for t in list(groups.keys()):
        d1 = h.create_group(t)
        for cond in s_conds:
            d2 = d1.create_group(f'{cond}_strf')
            for lvl in lv:
                d3 = d2.create_dataset(
                    lvl, data=groups[t][f'{cond}_strf'][lvl])

        for cond in nnc.conds:
            d2 = d1.create_group(cond)
            for lvl in lv:
                d3 = d2.create_dataset(lvl, data=groups[t][cond][lvl])

        for cond in nns.conds:
            d2 = d1.create_group(cond)
            for pt in p_type:
                d3 = d2.create_group(pt)
                for lvl in lv:
                    d4 = d3.create_dataset(lvl, data=groups[t][cond][pt][lvl])

