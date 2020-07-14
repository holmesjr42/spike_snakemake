import numpy as np
import h5py
import spike_tools as st
import pickle
import notch_gap_params as p_tool
import parameters as pm
import sys

com = pm.Commonfeat  # common features
nnc = pm.Nnmcontfeat  # continuous notched noise features
nns = pm.Nnsifeat  # notch on-off features

lfpf = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
      '/dataset/lfp_fil/cr50_190724_lfp_fil.hdf5'

tspf = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
      '/dataset/tsp/cr50_190724_tsp.hdf5'

p_info = com.p_info
freq = com.list_freq
cond = 'nnm_3_0'
strf = 'strf_4'
t_w = [10, 31]
bfs = 7336

# strf
oct_r = p_tool.o_cut(bfs, option=cond)
# octave rage calculation
freq_o = np.around(np.log2(freq / bfs), decimals=2)
list_o = freq_o[oct_r[0]:oct_r[1] + 1]

with h5py.File(tspf, 'r') as h_5:

    time_snip = h_5[strf]
    print(np.shape(time_snip))
    # mua = st.index_strf(time_snip)
    # tsps = st.tsp_strf(p_info, mua, twi=t_w[0], twf=t_w[1])
    # ttsp = np.array(tsps[0])
    # atsp = np.array(tsps[1])
    # a_t = ttsp[:, oct_r[0]:oct_r[1] + 1, :]
    # a_f = atsp[:, oct_r[0]:oct_r[1] + 1]

with h5py.File(lfpf, 'r') as l_5:
    # print(animal, s_cord, i)
    lfp = l_5[strf]
    print(np.shape(lfp))
    # lfps = st.lfp_strf(p_info, lfp)
    # llfp = np.array(lfps[0])
    # a_l.append(llfp[:, oct_r[0]:oct_r[1] + 1, :])