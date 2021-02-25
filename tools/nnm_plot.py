import os
import sys

sys.path.insert(1, os.path.dirname(os.getcwd()))
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
from parameters import parameters as pm
import seaborn as sns
import h5py
from matplotlib.colors import ListedColormap

sys.path.pop(1)


def animal_name(file_input):
    name = file_input.split('/')[-1].split('_')[0]
    return name


def pop_lvl(animal, data, cond):
    cat = list(data.keys())
    if 'all' in animal:
        for i in cat:
            for j in cond:
                if '30' in list(data[cat[0]].keys()):
                    data[cat[i]][j].pop('30')

                else:
                    pass

    else:
        print('nothing to pop')
        pass

    return data


# function for nnm plotting
def nnm_plot(aax, dats, imag, cm):
    aax.tick_params(axis='both', which='major', labelsize=8, length=3,
                    width=0.7)
    # aax.tick_params(axis='both', length=0.9, width=0.5)
    aax.spines['top'].set_visible(False)
    aax.spines['right'].set_visible(False)
    aax.spines['left'].set_visible(False)
    aax.spines['bottom'].set_visible(False)
    # aax.spines['bottom'].set_linewidth(0.5)
    # aax.spines['left'].set_linewidth(0.5)
    for tick in aax.get_xticklabels():
        tick.set_fontname('sans-serif')
    for tick in aax.get_yticklabels():
        tick.set_fontname('sans-serif')
    img = aax.imshow(
        dats, cmap=cm, aspect=100 / 3,
        extent=[0, 100, 1.5, -1.5], interpolation='gaussian')
    imag.append(img)
    # plt.subplots_adjust(wspace=.001)
    aax.set_xticks(np.linspace(0, 100, 5, endpoint=True), minor=True)
    aax.set_xticks(np.linspace(0, 50, 2, endpoint=True))
    # aax.set_yticks([0, 4, 8, 12, 16, 20, 24])
    aax.set_yticks([-1.5, -1., -0.5, 0., 0.5, 1., 1.5], minor=True)
    aax.set_yticks([-1., 0., 1.])

    # axins = inset_axes(aax,
    #                    width="5%",  # width = 5% of parent_bbox width
    #                    height="100%",  # height : 50%
    #                    loc='lower left',
    #                    bbox_to_anchor=(20, 0., 1, 1),
    #                    bbox_transform=aax.transAxes,
    #                    borderpad=0,
    #                    )
    aax.label_outer()
    # return aax, axins


# normalize the heatmap


def norm_color(img, types, ratio):
    """
    Normalize the heatmap for firing rate/lfp
    :param img: img list from plotting. Check individual nnm_*_sp_lfp script
    and search img
    :param types: type of the data. Either 'lfp' or 'tsp'
    :param ratio: ratio to adjust the max/min value for the heatmap
    :return: max value of the heatmap for the spike colorbar
    """

    if types == 'lfp':
        vmin = min(ims.get_array().min() for ims in img) * ratio
        vmax = -1 * vmin
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    elif types == 'tsp':
        vmin = 0
        vmax = max(ims.get_array().max() for ims in img) * ratio
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    else:
        print('there is a problem with norm_color. check it')
        sys.exit()

    for ims in img:
        ims.set_norm(norm)

    return vmax


def sub_plots(fig, row, col, n, ws, hs):
    """
    :param fig: figure
    :param row: number of row (e.g. number of channels)
    :param col: number of columns (e.g. 2 for NN/Ctrl case)
    :param n: order of the subplot (e.g. first subplot n = 0)
    :param ws: width space
    :param hs: height space
    :return: subplot columns
    """
    sub_fig = fig.subplots(row, col,
                           gridspec_kw=dict(
                               left=0.12 + 0.185 * n,
                               right=0.286 + 0.185 * n,
                               top=0.90, bottom=0.07,
                               wspace=ws, hspace=hs
                               ))

    return sub_fig


def nnm_labels(aaxs, notch, lvp, depth, ch, j, i):
    """
    This is the function specifically for minor plot setup for heatmap strf
    part
    :param aaxs: list of axes for notch, control, or white noise
    :param notch: width of the notch
    :param lvp: SPL list you are using for this plot
    :param depth: depth from parameters/parameters
    :param ch: channel list from parameters/ch
    :param j: order of lvp for loop
    :param i: order of len(ch) for loop
    :return:
    """
    if len(aaxs) == 2:
        # lines for the gap
        for axes in aaxs:
            for p in [-notch, notch]:
                axes.axhline(y=p, color='white', linestyle='-.', lw=0.6)
            # aaxc.axvline(0.2, 0, 1, label='pyplot vertical line')

        # y lables
        if j == lvp[0]:
            aaxs[1].set_ylabel(depth[i] + r'$\mu$' + 'm' + '\n\n\n\n\n\n ',
                               fontsize=10)
        else:
            pass

        # x label
        if i == len(ch) - 1:
            aaxs[1].text(1, -0.4, 't (ms)',
                         horizontalalignment='center',
                         verticalalignment='center',
                         # rotation='vertical',
                         transform=aaxs[1].transAxes,
                         fontdict={'weight': 'bold', 'size': 11}
                         )
        # subplot title
        elif i == 0:
            aaxs[0].title.set_text('NN')
            aaxs[1].title.set_text('Control')
        else:
            pass


def colorbar_set(fig, axs, vmax, lv, lvi, img, types):
    """
    This is a function for minor colorbar setups
    :param fig: name of the main figure (e.g. c1_fig)
    :param axs: dic of the axs (e.g. c1_axs)
    :param cmx: the maximum of the colorbar
    :param lv: SPL of the current loop for lvp
    :param lvi: the first element of lvp. Just to pick the coordination of
    fig text
    :param img: img list for current loop heatmap setups. Check c_img or l_img
    from nnm_*_sp_lfp script
    :param type: the type of the data. Either 'lfp' or 'tsp'
    :return:
    """
    if types == 'lfp':
        ticklist = [-vmax, 0, vmax]
    elif types == 'tsp':
        ticklist = np.linspace(0, vmax, 3, endpoint=True)
    else:
        print('you did not choose the data type for colorbar_set.')
        sys.exit()

    cbart = fig.colorbar(
        img, ax=axs[lv],
        orientation='horizontal',
        format='%.2f',
        fraction=.015,
        ticks=ticklist
        )

    cbart.ax.tick_params(labelsize=10)

    if types == 'lfp':
        cbart.ax.set_title('Amplitude ' + r'(mV)',
                           fontdict={'fontsize': 11})
    elif types == 'tsp':
        cbart.ax.set_title('Firing rate (spk/s)',
                           fontdict={'fontsize': 11})
    else:
        print('you did not choose the data type for colorbar_set.')
        sys.exit()

    cbart.update_ticks()

    axs[lv][0, 0].text(1.3, 1.5, f'{lv} SPL dB',
                       horizontalalignment='center',
                       verticalalignment='center',
                       # rotation='vertical',
                       transform=axs[lv][0, 0].transAxes,
                       fontdict={'weight': 'bold', 'size': 11}
                       )

    if lv == lvi:
        axs[lv][2, 0].text(-1.1, 0.5, 'Depth\n\n\n\n\n\n\nOctave',
                           horizontalalignment='center',
                           verticalalignment='center',
                           rotation='vertical',
                           transform=axs[lv][2, 0].transAxes,
                           fontdict={'weight': 'bold', 'size': 11}
                           )
    else:
        pass
