import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))

from tools import icsd
import numpy as np
from parameters import parameters as pm
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import quantities as pq
import neo.core

sys.path.pop(1)

'''This is for a single animal data. CSD for nnm continuous condition. Best 
frequeny at the high edge of the gap. Following traditional way with 
dividing with h**2 and no sigma is used. Uses rappel filter (it's hamming 
filter but has different parameters from genenral hamming window) 

# used icsd.py to get raw csd. If you use estimated_csd or elephant,
# which gives you the output of estimated_csd, you will have filtered csd
# instead of raw one, which is different from what your supervisor asked you
# to calculate. But maybe it's not bad to study how to get estimated_csd.
# for this one, I used traditional CSD filtered with hamming three point
# method. '''

'''
This is for a single animal data
'''
sns.set_style("ticks")
val = pm.Setup(cond='nnm_3_1', cond2='tsp', mode=1, ccmap='csd')
ccmap = val.values()[-1]
# ccmap = 'jet_r'
file_input = snakemake.input[0]
# file_input = '/home/bhc/OneDrive/Work/PhD/projects/zwicker_tone_2018_2021' \
#              '/data/dataset/nn_lfp_avg/cr50_190724_nn_lfp_a.pk'

animal = file_input.split('/')[-1].split('_')[0]
sr_lfp = 1061.5 * pq.Hz  # for convert time into data coord in the data file
ons = [0, 500]  # time window for onset response
offs = [1000, 1500]  # time window for offset response
on_p = pm.lfp_p(ons)  # convert onset time window range into the data coord
off_p = pm.lfp_p(offs)  # convert offset time window range into the data coord

with open(file_input, 'rb') as dat:
    datas = pickle.load(dat)



# la[0] is notched noise and la[1] is white noise
la = np.array([datas['la']['nn_1_1']['70'], datas['la']['nn_wn_1_1']['70']
               ]) * pq.microvolt

la.units = pq.mV
# converting lfp data format to neo analogsignal to use it with icsd.py
lfp_data1 = neo.core.AnalogSignal(la[0].T, sampling_rate=sr_lfp)
lfp_data2 = neo.core.AnalogSignal(la[1].T, sampling_rate=sr_lfp)
coord = np.arange(0, 1501, 100) * pq.micrometer
sigma = 1 / (100 * pq.micrometer)

'''
I don't know how this works, but the  approximation of raw csd is (V[z-h] 
-2 * V[z]**2 + V[z+h])/h ** 2, which removes the conductance out of the 
formula. So instead of using the default sigma, I changed the value to 
1/h, because in icsd.py, h is divided only one time.
'''
# input args for standard CSD function in iCSD.py
std_input_n = {
    'lfp': lfp_data1,
    'coord_electrode': coord,
    'sigma': sigma,
    'f_type': 'gaussian',
    'f_order': (3, 1)
    }

std_input_w = {
    'lfp': lfp_data2,
    'coord_electrode': coord,
    'sigma': sigma,
    'f_type': 'gaussian',
    'f_order': (3, 1)
    }

std1 = icsd.estimate_csd(**std_input_n)
rn_csd = np.transpose(std1[0] / std1[0].units) * std1[0].units
fn_csd = np.transpose(std1[1] / std1[1].units) * std1[1].units
rn_csd.units = pq.mV / (pq.mm ** 2)
fn_csd.units = pq.mV / (pq.mm ** 2)

std2 = icsd.estimate_csd(**std_input_w)
rw_csd = np.transpose(std2[0] / std2[0].units) * std2[0].units
fw_csd = np.transpose(std2[1] / std2[1].units) * std2[1].units
rw_csd.units = pq.mV / (pq.mm ** 2)
fw_csd.units = pq.mV / (pq.mm ** 2)

'''
plot setup
'''


def csd_plot(zz, ax, image, sec='whole', cmap=ccmap):
    zz = zz / zz.units
    if sec == 'whole':
        pi = 0
        pf = np.shape(zz)[1]
        ti = 0
        tf = 2100
        t_ticks = np.linspace(0, 2, 9, endpoint=True)
        t_ticks2 = np.linspace(0, 2, 5, endpoint=True)
    elif sec == 'onset':
        pi = on_p[0]
        pf = on_p[1] + 1
        ti = ons[0]
        tf = ons[1]
        t_ticks = np.linspace(ons[0] * 1e-3, ons[1] * 1e-3, 11, endpoint=True)
        t_ticks2 = np.linspace(ons[0] * 1e-3, ons[1] * 1e-3, 6, endpoint=True)
    elif sec == 'offset':
        pi = off_p[0]
        pf = off_p[1] + 1
        ti = offs[0]
        tf = offs[1]
        t_ticks = np.linspace(offs[0] * 1e-3, offs[1] * 1e-3, 11,
                              endpoint=True)
        t_ticks2 = np.linspace(offs[0] * 1e-3, offs[1] * 1e-3, 6,
                               endpoint=True)
    else:
        print('you did not choose section for csd.')
        sys.exit()
    sig = zz[:, pi:pf]
    img = ax.imshow(
        sig, cmap=cmap, aspect=(pf - pi) * 1e-3 / 1500,
        extent=[ti * 1e-3, tf * 1e-3, -1500, 0], interpolation='gaussian')
    image.append(img)
    yticks = np.arange(-1500, 1, 100)
    yticks2 = np.arange(-1500, 1, 300)
    ax.set_yticks(yticks2)
    ax.set_yticks(yticks, minor=True)
    ax.set_xticks(t_ticks, minor=True)
    ax.set_xticks(t_ticks2)

    if sec == 'whole':
        axins = inset_axes(ax,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.1, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )
    else:
        axins = inset_axes(ax,
                           width="5%",  # width = 5% of parent_bbox width
                           height="100%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.1, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0,
                           )

    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.tick_params(axis='y', which='major', labelsize=9)

    return ax, axins


'''
plots
'''
f_csd = [fn_csd, fw_csd]
cond = ['onset', 'onset', 'offset', 'offset']
dats = [0, 1, 0, 1]
el_img = []
pl_img = []
ec_img = []
pc_img = []


def cbar_t(cbar, type):
    cbar.ax.tick_params(labelsize=9)
    if type == 'lfp':
        cbar.ax.set_title(
            r'mV',
            fontsize=9
            )
    elif type == 'csd':
        cbar.ax.set_title(
            r'mV/$mm^{2}$',
            fontsize=9
            )

    # cbar.uptade_ticks()
    # cbar.ax.text(-10, 4.4, r'$\times$10$^{-3}$' + '  ' + r'mV/$mm^{2}$',
    #              ha='center', va='top', rotation='horizontal', fontsize=8)


# the whole section
e_fig, e_axes = plt.subplots(2, 2, figsize=(9, 9), dpi=300)
e_fig.subplots_adjust(top=0.95, bottom=0.15, hspace=0.4)
cbar_el = 0
cbar_ec = 0

for i in range(2):
    ax_el, axins_el = csd_plot(la[i], e_axes[0, i], el_img, sec='whole')
    ax_ec, axins_ec = csd_plot(f_csd[i], e_axes[1, i], ec_img, sec='whole')

    if i == 1:
        cbar_el = plt.colorbar(el_img[i], cax=axins_el,
                               orientation='vertical', format='%.2f')
        cbar_ec = plt.colorbar(ec_img[i], cax=axins_ec,
                               orientation='vertical', format='%.2f')
        cbar_t(cbar_el, 'lfp')
        cbar_t(cbar_ec, 'csd')
    else:
        axins_el.remove()
        axins_ec.remove()
    e_axes[1, i].set_xlabel('t [s]')
    e_axes[i, 0].set_ylabel('Depth' + ' [' + r'$\mu$' + 'm' + ']')

e_axes[0, 0].set_title('2 s Notched Noise')
e_axes[0, 1].set_title('2 s White Noise')
e_axes[0, 0].text(-0.75, -750, 'LFP', ha='center', va='center',
                  fontsize=15, rotation=90)
e_axes[1, 0].text(-0.75, -750, 'CSD', ha='center', va='center',
                  fontsize=15, rotation=90)

# onset/offset sections
p_fig, p_axes = plt.subplots(2, 4, figsize=(16, 9), dpi=300)
p_fig.subplots_adjust(left=0.1, top=0.97, hspace=0.2, wspace=0.35)
cbar_pl = 0
cbar_pc = 0

for i in range(4):
    d = dats[i]
    ax_pl, axins_pl = csd_plot(la[d], p_axes[0, i], pl_img, sec=cond[i])
    ax_pc, axins_pc = csd_plot(f_csd[d], p_axes[1, i], pc_img, sec=cond[i])

    p_axes[1, i].set_xlabel('t [s]')
    if i == 0 or i == 1:
        p_axes[i, 0].set_ylabel('Depth' + ' [' + r'$\mu$' + 'm' + ']')
    if i == 1 or i == 3:
        cbar_pl = plt.colorbar(pl_img[i], cax=axins_pl,
                               orientation='vertical', format='%.2f')
        cbar_pc = plt.colorbar(pc_img[i], cax=axins_pc,
                               orientation='vertical', format='%.2f')
        cbar_t(cbar_pl, 'lfp')
        cbar_t(cbar_pc, 'csd')
    else:
        axins_pl.remove()
        axins_pc.remove()

p_axes[0, 0].set_title('Onset Notched Noise')
p_axes[0, 1].set_title('Onset White Noise')
p_axes[0, 2].set_title('Offset Notched Noise')
p_axes[0, 3].set_title('Offset White Noise')

p_axes[0, 0].text(-0.2, -750, 'LFP', ha='center', va='center',
                  fontsize=15, rotation=90)
p_axes[1, 0].text(-0.2, -750, 'CSD', ha='center', va='center',
                  fontsize=15, rotation=90)


# normalize
def norm_color(img, types, ratio=0.6):
    if types == 'lfp':
        # vmin = min(ims.get_array().min() for ims in img) * ratio
        # vmax = max(ims.get_array().max() for ims in img) * ratio
        vmax = max(max(ims.get_array().max() for ims in img), np.abs(min(
            ims.get_array().min() for ims in img))) * ratio
        vmin = (-1) * vmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif types == 'csd':
        vmax = max(max(ims.get_array().max() for ims in img), np.abs(min(
            ims.get_array().min() for ims in img))) * ratio
        vmin = (-1) * vmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        print('there is a problem with norm_color. check it')
        sys.exit()

    for ims in img:
        ims.set_norm(norm)


norm_color(pl_img[0:2], 'lfp')
norm_color(pc_img[0:2], 'csd')
norm_color(pl_img[2:4], 'lfp')
norm_color(pc_img[2:4], 'csd')
norm_color(el_img, 'lfp')
norm_color(ec_img, 'csd')
for i in [cbar_el, cbar_ec, cbar_pl, cbar_pc]:
    i.ax.locator_params(nbins=5)


# part of them

e_fig.savefig(snakemake.output.whole, dpi=300)
p_fig.savefig(snakemake.output.part, dpi=300)

# p_fig.savefig('testing_1.png', dpi=300)
# e_fig.savefig('testing_2.png', dpi=300)
# plt.show()
