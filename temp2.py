import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# '''
# This is for a single animal data
# '''
# sr_lfp = 1061.5  # lfp sampling rate
#
# file_path = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#             '/data/dataset/'
#
# animal = 'cr50_190724'
#
# d_lfp = file_path + 'lfp_fil/' + animal + '_lfp_fil.hdf5'
# d_spk = file_path + 'tsp/' + animal + '_tsp.hdf5'
#
# # k = neo.core.RecordingChannel
# f = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data' \
#     '/dataset/nn_lfp_avg/cr50_190724_nn_3_1_lfp_a.pk'
# with open('/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021/data'
#           '/dataset/nn_lfp_avg/cr50_190724_nn_3_1_lfp_a.pk', 'rb') as dat:
#     datas = pickle.load(dat)
#
# la = datas['la']['nn_3_1']
#
# # df = pd.DataFrame(la)
# # df.to_csv('matlab.csv')
# coord = np.arange(0, 1501, 100) * pq.micrometer
# csds = []
# al = np.transpose(la)
#
# for i in range(al.shape[0]):
#     als = al[i]
#     cs = []
#     for j in np.arange(1, 15, 1):
#         p = (als[j - 1] - (2 * als[j]) + als[j + 1]) / (0.0001**2)
#         # print(p)
#         cs = np.append(cs, p)
#     csds.append((-1*cs)/1000000)
#
#
# # elephant.current_source_density.estimate_csd(la, method='KCSD1D')
# al = np.transpose(la)
# sig = AnalogSignal(al * pq.mV, sampling_rate=sr_lfp * pq.Hz)
#
# # chx = ChannelIndex(name='Probe 0', index=range(16), channel_ids=range(16),
# #                    channel_names=['Channel %i' % chid for chid in range(
# #                        16)], coordinates=coord * pq.micrometer)
# # chx.analogsignals.append(sig)
# # sig.channel_index = chx
#
# print(sig.shape[1])
# hey = elephant.current_source_density.estimate_csd(lfp=sig,
#                                                    coords=coord,
#                                                    sigma=1 * pq.S / pq.m,
#                                                    method='StandardCSD')
#
# # #
# # plt.figure()
# # t = np.arange(0, 16)
# # plt.plot(t[1:15], csds[3500], color='black')
# # plt.plot(t, hey[3500]*(10**3))
# # plt.show()
# # # # hello = neo.io.NixIO('hey.h5')
# # # # hello.write_analogsignal(hey)
# # # # with open('hello.pkl', 'wb') as hu:
# # # #     pickle.dump(hey, hu, protocol=pickle.HIGHEST_PROTOCOL)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6, 3])

axins1 = inset_axes(ax1,
                    width="50%",  # width = 50% of parent_bbox width
                    height="5%",  # height : 5%
                    loc='upper right')

im1 = ax1.imshow([[1, 2], [2, 3]])
fig.colorbar(im1, cax=axins1, orientation="horizontal", ticks=[1, 2, 3])
axins1.xaxis.set_ticks_position("bottom")

axins = inset_axes(ax2,
                   width="5%",  # width = 5% of parent_bbox width
                   height="50%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax2.transAxes,
                   borderpad=0,
                   )

# Controlling the placement of the inset axes is basically same as that
# of the legend.  you may want to play with the borderpad value and
# the bbox_to_anchor coordinate.

im = ax2.imshow([[1, 2], [2, 3]])
fig.colorbar(im, cax=axins, ticks=[1, 2, 3])

plt.show()
