import numpy as np

# import scipy.fftpack as ft

'''
spike train

 - making binary spike train array
 - time_snip is Tsp in mat file
'''


def index_strf(time_snip):
    mua = []
    for i in range(16):
        mua.append(np.zeros(181*1000 + 1))
        for k in range(np.size(time_snip[i])):
            k1 = int(np.round(1000 * time_snip[i][k]))
            if k1 > 181*1000:
                break
            elif k1 > 0:
                mua[i][k1] = 1
    return mua


'''
tsp

averaging the number of spikes
 - p_info = parameter_info = time_index; mua, spike train binary arrays from
 - index_strf; twi, ms, initial point of time window for activated area in
 - 100 ms
 - window from a stimulus trigger point; twf, ms, final point of time window
 - from twi; tsp_m, spks,  average of each frequency activity in 100 ms time
 - window;,
 - avg, spike density function, average of each frequency activity through 
   activation time window
'''

def tsp_strf(p_info, mua, twi=19, twf=50):
    tsp_m = [[] for _ in range(16)]
    tsp_f = [[] for _ in range(16)]
    avg = [[] for _ in range(16)]
    for i in range(16):
        tsp_m[i] = [[] for _ in range(49)]
        tsp_f[i] = [[] for _ in range(49)]
        for j in range(49):
            for k in range(len(p_info[j]) - 1):
                tsp_f[i][j].append(mua[i][(int(np.floor(p_info[j][k]))):(int(
                    np.floor(p_info[j][k])) + 101)])

            tsp_m[i][j] = np.append(tsp_m[i][j],  np.mean(tsp_f[i][j], 0))
            avg[i] = np.append(avg[i], np.mean(tsp_m[i][j][twi:twf]))

    return [tsp_m, avg]


'''
lfp

 - averaging lfp
 - p_info = parameter_info = time_index; mua, spike train binary arrays from
 - index_strf; twi, ms, initial point of time window for activated area in
 - 100 ms
 - window from a stimulus trigger point; twf, ms, final point of time window
 - from twi; tsp_m, mV,  average of each frequency voltage in 100 ms time
 - window;,
 - avg, mV/ms, average of each frequency activity through activation time 
   window. What do I call them? Voltage density function?
'''


def lfp_strf(p_info, lfp, twi=19, twf=50):

    sr_lfp = 1061.5
    lfp_m = [[] for _ in range(16)]
    lfp_f = [[] for _ in range(16)]
    avg = [[] for _ in range(16)]
    for i in range(16):
        lfp_m[i] = [[] for _ in range(49)]
        lfp_f[i] = [[] for _ in range(49)]
        for j in range(49):
            for k in range(len(p_info[j]) - 1):
                cut_l = int(np.round(p_info[j][k] * (sr_lfp/1000)))
                cut_h = int(np.round((p_info[j][k] + 101) * (sr_lfp/1000)))
                lfp_f[i][j].append(lfp[i][cut_l:cut_h])

            lfp_m[i][j] = np.append(lfp_m[i][j],  np.mean(lfp_f[i][j], 0))
            avg[i] = np.append(avg[i], np.mean(lfp_m[i][j][twi:twf]))

    return [lfp_m, avg]


