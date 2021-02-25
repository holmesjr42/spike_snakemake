import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))

import matplotlib.pyplot as plt
import scipy.signal as scig
import numpy as np
# import spike_tools as st
from parameters import parameters as pm
import sys

sys.path.pop(1)

# time index for notched noise. in[0] is the lower boundary and in[1] is
# the higher boundary.
in31 = pm.Nn.nn_i['nn31']
in11 = pm.Nn.nn_i['nn11']
val = pm.Setup(cond='nnm_3_0', cond2='tsp', mode=1)
crl = val.values()[7]
sr_lfp = 1061.5  # lfp sampling rate

# smoothing filter using hanning window
filt = np.hanning(5)


def index_spike(tsp):
    """
    :param tsp: just raw mua data from tsp file
    :return: spike train
    """
    mua = []
    for i in range(16):
        mua.append(np.zeros(181 * 1000 + 1))
        for k in range(np.size(tsp[i])):
            k1 = int(np.round(1000 * tsp[i][k]))
            if k1 > 181 * 1000:
                break
            elif k1 > 0:
                mua[i][k1] = 1
    return mua


def psth(mua, indt):
    """
    :param mua: spike train from index_spike
    :param indt: type of index. depending on 3s/1s or 1s/1s on-off condition.
    :return: n_raster: data array you can use for raster plot. each channel
    has 4 s spike train with certain repetition ( 3s/1s # 45, 1s/1s # 90)
             n_psth: data array for psth. It's the sum of all repetition.
             s_psth: filtered data. Used convolution moving average, hanning
                     window for smoothing and averaged with repetition number.
             t_n: time array for generating plot.
    """
    n_raster = [[] for _ in range(16)]
    n_psth = []
    s_psth = []

    if indt == 'nn_3_1' or indt == 'nn_wn_3_1':
        ind = in31
        t_n = np.arange(0, 4001)

    elif indt == 'nn_1_1' or indt == 'nn_wn_1_1':
        ind = in11
        t_n = np.arange(0, 2001)

    else:
        print('somthing is wrong. check if you choose the right index type '
              'for nn condition.')
        sys.exit()

    rep = np.size(ind[0])

    for i in range(16):
        n_raster[i] = []

        for j in range(rep):  # session of on nn31 file with 4000 ms/session
            n_raster[i].append(mua[i][ind[0][j]:ind[1][j] + 1])
        n_psth.append(np.sum(n_raster[i], axis=0))
        s_psth.append(np.convolve(filt / filt.sum(), n_psth[i] / rep,
                                  mode='same'))
    return n_raster, n_psth, s_psth, t_n


def lfp_seg(lfp, indt):
    l_seg = [[] for _ in range(16)]
    rep = 0

    if indt == 'nn_3_1' or indt == 'nn_wn_3_1':
        ind = in31

    elif indt == 'nn_1_1' or indt == 'nn_wn_1_1':
        ind = in11
    else:
        print('something is wrong with choosing time index.')
        sys.exit()

    rep = np.size(ind[0])
    ind_l = ind[0] * (sr_lfp / 1000)
    ind_h = (ind[1] + 101) * (sr_lfp / 1000)
    for i in range(16):
        for j in range(rep):
            l_seg[i].append(lfp[i][np.int(ind_l[j]):
                                   np.int(ind_h[j])])
    return l_seg


def lfp_index(t1, t2=0, types=0):
    if types == 0:
        inds = np.int(np.round(t1 * (sr_lfp / 1000)))
    elif types == 1:
        inds = int(np.round(
            t1 * (sr_lfp / 1000) + int((t2 * sr_lfp / 1000))
            ))
    else:
        print("you didn't choose the right types")
        raise
    return inds


def m_spectro(l_seg):
    """

    :param l_seg: data array from lfp_seg
    :return: f: frequency array from spectrogram
             t: time array from spectrogram
             sx_mean: mean value of spectrogram intensity
             sx_m_log: mean value of spectrogram log intensity
    """

    sx_mean = []
    f = []
    t = []
    sx_m_log = []
    for i in range(len(l_seg)):
        sx_a = []
        sx_l = []
        rep = len(l_seg[i])
        for j in range(rep):
            f1, t1, sxr = scig.spectrogram(l_seg[i][j], sr_lfp,
                                           window=scig.hann(172), nperseg=172)
            sx = (sxr - np.min(sxr)) / (np.max(sxr) - np.min(sxr))  # scaling
            sxl = (np.log10(sxr + 1) - np.min(np.log10(sxr + 1))) / \
                  (np.max(np.log10(sxr + 1)) - np.min(np.log10(sxr + 1)))
            # scaling log
            sx_a.append(sx)
            sx_l.append(sxl)
            if j == rep - 1:
                f.append(f1)
                t.append(t1)

        sx_mean.append(np.mean(sx_a, axis=0))
        sx_m_log.append(np.mean(sx_l, axis=0))
    return f, t, sx_mean, sx_m_log


def spect_plot(ax, t, f, sig, images, xticks, yticks):
    im = ax.pcolormesh(t, f, sig,
                       cmap='rainbow',
                       edgecolors='none', shading='flat')
    images.append(im)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_yticks(yticks)
    ax.set_xticks(xticks[0])
    # ax.set_yticks(yticks[1], minor=True)
    ax.set_xticks(xticks[1], minor=True)
    ax.tick_params(axis='both', which='major', labelsize=8, length=2.5,
                   width=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=8, length=0.9,
                   width=0.25)


def psth_plot(ax, t, spikes, xticks, yticks, c_type):
    """

    :param ax: axis of a plot/subplo
    :param t: time array
    :param spikes: spike number array
    :param xticks: array for the xtick
    :param yticks: array for the ytick
    :param s_max: maximum value of the spike number in all 16 channels
    :param c_type: 'nn' or 'wn' to choose the line color
    :return:
    """

    if c_type == 'nn_3_1' or c_type == 'nn_1_1':
        c_line = crl[0]
    elif c_type == 'nn_wn_3_1' or c_type == 'nn_wn_1_1':
        c_line = crl[1]
    else:
        print('something is wrong with psth_plot in nn_tools')
        sys.exit()

    ax.plot(t / 1000, spikes, lw=1, color=c_line, alpha=0.8)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)


def lfp_plot(ax, t, lfp, conf, xticks, yticks, c_type):
    """
    :param ax: axis of a plot/subplo
    :param t: time array
    :param lfp: averaged lfp
    :param conf: confidence interval of lfp
    :param xticks: array for the xtick
    :param yticks: array for the ytick
    :param c_type: 'nn' or 'wn' to choose the line color
    :return:
    """

    if c_type == 'nn_3_1' or c_type == 'nn_1_1':
        c_line = crl[0]
    elif c_type == 'nn_wn_3_1' or c_type == 'nn_wn_1_1':
        c_line = crl[1]
    else:
        print('something is wrong with lfp_plot nn_tools')
        sys.exit()
    lfp = lfp
    conf1 = np.transpose(conf)

    ax.plot(t / 1000, lfp, lw=1, color=c_line, alpha=0.8)
    ax.fill_between(
        t / 1000,
        conf1[0],
        conf1[1],
        alpha=.2
        )
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)


def setups(cond):
    if cond == 'nn_3_1':
        names = ['nn_3_1', 'nn_wn_3_1']
        tmax = 4100
        nn_title = 'NN On 3 s/ Off 1 s'
        wn_title = 'WN On 3 s/ Off 1 s'

    elif cond == 'nn_1_1':
        names = ['nn_1_1', 'nn_wn_1_1']
        tmax = 2100
        nn_title = 'NN On 1 s/ Off 1 s'
        wn_title = 'WN On 1 s/ Off 1 s'
    else:
        print('something is wrong. check nn31/11_psth_lfp_plot.py')
        sys.exit()
    return names, tmax, nn_title, wn_title
