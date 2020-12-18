'''
qmodisplay.py

version 1.1
last updated: December 2020

by Trevor Arp
Quantum Materials Optoelectronics Laboratory
Department of Physics and Astronomy
University of California, Riverside, USA

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Description:
This is generic visualization code, modified from the QMO Lab's internal code libraries,
as such all or part of this code may be published elsewhere at some point in the future.

See accompanying README.txt for instructions on using this code.
'''

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

import numpy as np
import argparse

matplotlib.rcParams["keymap.fullscreen"] = ''

class figure_inches():
    '''
    A class to assist in the layout of figures, handling the conversion from matplotlibs
    0 to 1 units to actual physical units in inches. Will also generate figures and axes
    of various kinds from specifications given in inches.

    Args:
        name (str) : The name of the figure, can't have two figures with the same name in a script
        xinches (float) : The width of the figure in inches
        yinches (float) : The height of the figure in inches
    '''

    def __init__(self, name, xinches, yinches):
        self.xinches = xinches
        self.yinches = yinches
        self.r = yinches/xinches
        self.name = name
        self.fig = plt.figure(self.name, figsize=(self.xinches, self.yinches), facecolor='w')
    # end init

    def get_fig(self):
        '''
        Returns a figures object with the physical size given
        '''
        return self.fig
    # end make_figure

    def make_axes(self, spec, zorder=1):
        '''
        Makes and returns a matplotlib Axes object.

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        plt.figure(self.name)
        return plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder)
    # make_axes

    def make_3daxes(self, spec, zorder=1):
        '''
        Makes and returns a matplotlib Axes object with a 3D projection

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        plt.figure(self.name)
        return plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches], zorder=zorder, projection='3d')
    # make_3daxes

    def make_dualy_axes(self, spec, color_left='k', color_right='k', zorder=1, lefthigher=True):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis.

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_left (str, optional) : The color (in matplotlib notation) of the left y-axis, default black.
            color_right (str, optional) : The color (in matplotlib notation) of the left x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
            lefthigher (bool, optional) : If True (default) the left axis will be on top and provide the x-axis.
        '''
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        if lefthigher:
            zorderl = zorder + 1
            zorderr = zorder
        else:
            zorderl = zorder
            zorderr = zorder + 1
        #

        axl = plt.axes([spec[0]+1, spec[1]+1, spec[2]+1, spec[3]+1], zorder=zorderl)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.tick_params('y', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)

        axr = plt.axes([spec[0]+2, spec[1]+2, spec[2]+2, spec[3]+2], zorder=zorderr)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', colors=color_right)
        return axl, axr
    # make_dualy_axes

    def make_dualx_axes(self, spec, color_bottom='k', color_top='k', zorder=1):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis. Note, the
        first axes returned (the bottom x-axis) is "on top" and provides the y-axis

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_bottom (str, optional) : The color (in matplotlib notation) of the bottom x-axis, default black.
            color_top (str, optional) : The color (in matplotlib notation) of the top x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.
        '''
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axb = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder+1)
        axb.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axb.patch.set_alpha(0)
        axb.tick_params('x', colors=color_bottom)
        axb.spines['bottom'].set_color(color_bottom)
        axb.spines['top'].set_color(color_top)

        axt = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder)
        axt.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axt.patch.set_alpha(0)
        axt.yaxis.set_visible(False)
        axt.xaxis.tick_top()
        axt.xaxis.set_label_position("top")
        axt.tick_params('x', colors=color_top)
        return axb, axt
    # make_dualx_axes

    def make_dualxy_axes(self, spec, color_bottom='k', color_top='k', color_left='k', color_right='k', zorder=1):
        '''
        Makes and returns two overlaid axes, with two y axes sharing the same x-axis. Note: the
        first axis returned (the left y-axis) is "on top"

        Args:
            spec : A list of the dimensions of the axis [left, bottom, width, height] in inches
            color_bottom (str, optional) : The color (in matplotlib notation) of the bottom x-axis, default black.
            color_top (str, optional) : The color (in matplotlib notation) of the top x-axis, default black.
            color_left (str, optional) : The color (in matplotlib notation) of the left y-axis, default black.
            color_right (str, optional) : The color (in matplotlib notation) of the left x-axis, default black.
            zorder (int, optional) : The "z-axis" order of the axis, Axes with a higher zorder will appear
                on top of axes with a lower zorder.

        Returns:
            axes : Returns the axes in the following order: left, right, bottom, top
        '''
        plt.figure(self.name)
        ax0 = plt.axes([spec[0]/self.xinches, spec[1]/self.yinches, spec[2]/self.xinches, spec[3]/self.yinches])
        ax0.axis('off')

        axl = plt.axes([spec[0]+4, spec[0]+4, spec[0]+4, spec[0]+4], zorder=zorder+3)
        axl.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axl.patch.set_alpha(0)
        axl.xaxis.set_visible(False)
        axl.tick_params('y', colors=color_left)
        axl.spines['left'].set_color(color_left)
        axl.spines['right'].set_color(color_right)
        axl.spines['top'].set_color(color_top)
        axl.spines['bottom'].set_color(color_bottom)

        axr = plt.axes([spec[0]+3, spec[0]+3, spec[0]+3, spec[0]+3], zorder=zorder+2)
        axr.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axr.patch.set_alpha(0)
        axr.xaxis.set_visible(False)
        axr.yaxis.tick_right()
        axr.yaxis.set_label_position("right")
        axr.tick_params('y', colors=color_right)

        axb = plt.axes([spec[0]+1, spec[0]+1, spec[0]+1, spec[0]+1], zorder=zorder+1)
        axb.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axb.patch.set_alpha(0)
        axb.yaxis.set_visible(False)
        axb.xaxis.tick_bottom()
        axb.xaxis.set_label_position("bottom")
        axb.tick_params('x', colors=color_bottom)

        axt = plt.axes([spec[0]+2, spec[0]+2, spec[0]+2, spec[0]+2], zorder=zorder)
        axt.set_axes_locator(InsetPosition(ax0, [0.0, 0.0, 1.0, 1.0]))
        axt.patch.set_alpha(0)
        axt.yaxis.set_visible(False)
        axt.xaxis.tick_top()
        axt.xaxis.set_label_position("top")
        axt.tick_params('x', colors=color_top)
        return axl, axr, axb, axt
    # make_dualx_axes
# end figure_inches

def yaxis_right(ax):
    '''
    A simple function to move the y-axis of a matplotlib plot to the right.

    Args:
        ax : The matplotlib axes object to manipulate
    '''
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
# end yaxis_right

def xaxis_top(ax):
    '''
    A simple function to move the x-axis of a matplotlib plot to the top.

    Args:
        ax : The matplotlib axes object to manipulate
    '''
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
# end yaxis_right

def paper_figure_format(fntsize=12, font=None, bilinear=True, labelpad=0):
    '''
    Defines a standard format for paper figures and other production quality visualizations,
    can use any font that is in the maplotlib font folder Lib/site-packages/matplotlib/mpl-data/fonts/tff,

    Args:
        fntsize (int, optional) : The fontsize
        font (int, optional) : The font, in None (default) uses Arial
        bilinear (bool, optional) : If true (default) will use 'bilinear' interpolation option in imshow
        labelpad (float, optional) : The default padding between the axes and axes labels.
    '''
    matplotlib.rcParams.update({'font.family':'sans-serif'})
    if font is not None:
        matplotlib.rcParams.update({'font.sans-serif':font})
    else:
        matplotlib.rcParams.update({'font.sans-serif':'Arial'})
    matplotlib.rcParams.update({'font.size':fntsize})
    matplotlib.rcParams.update({'axes.labelpad': labelpad})
    matplotlib.rcParams.update({'axes.titlepad': labelpad})
    matplotlib.rcParams.update({'xtick.direction':'out'})
    matplotlib.rcParams.update({'ytick.direction':'out'})
    matplotlib.rcParams.update({'xtick.major.width':1.0})
    matplotlib.rcParams.update({'ytick.major.width':1.0})
    matplotlib.rcParams.update({'axes.linewidth':1.0})
    if bilinear:
        matplotlib.rcParams.update({'image.interpolation':'bilinear'})
# end paper_figure_format

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    '''
    Takes a color map and returns a new colormap that only uses the part of it between minval and maxval
    on a scale of 0 to 1

    Taken From: http://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib

    Args:
        cmap : The matplotlib colormap object to truncate
        minval (float, optional) : The minimum colorvalue, on a scale of 0 to 1
        maxval (float, optional) : The maximum colorvalue, on a scale of 0 to 1
        n (int, optional) : The number of samples, if -1 (default) uses the sampling from cmap
    '''
    if n == -1:
        n = cmap.N
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
# end truncate_colormap

def colorscale_map(darray, mapname='viridis', cmin=None, cmax=None, centerzero=False, truncate=None):
    '''
    Generates a Colormap, Normalization and ScalarMappable objects for the given data.

    Args:
        darray : The data, If color bounds not specified the min and max of the array are used,
            the mappable is initialized to the data.
        mapname : The name (matplotlib conventions) of the colormap to use.
        cmin (float, optional) : The minimum value of the map
        cmax (float, optional) : The maximum value of the map
        centerzero (bool, optional) : If true will make zero the center value of the colorscale, use
            for diverging colorscales.
        truncate (tuple, optional) : If not None (default) will truncate the colormap with (min, max)
            colorvalues on a scale of 0 to 1.

    Returns:
        Tuple containing (cmap, norm, sm) where cmap is the Colormap, norm is the
        Normalization and sm is the ScalarMappable.
    '''
    cmap = plt.get_cmap(mapname)
    if truncate is not None:
        cmap = truncate_colormap(cmap, truncate[0], truncate[1])
    if cmin is None:
        cmin = np.min(darray)
    if cmax is None:
        cmax = np.max(darray)
    if centerzero:
        cmax = max(np.abs([cmin, cmax]))
        cmin =  -1.0*cmax
    cNorm  = colors.Normalize(vmin=cmin, vmax=cmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    scalarMap.set_array(darray)
    return cmap, cNorm, scalarMap
# end colorscale_map

def change_axes_colors(ax, c):
    '''
    Changes the color of a given matplotlib Axes instance

    Note: for colorbars will need to use matplotlib.rcParams['axes.edgecolor'] = c before
    calling colorbarBase because whoever coded that clase bound the colorbar axes to that
    param value rather than having it function like a normal axis, which it is.

    Args:
        ax : The matplotlib axes object to manipulate
        c : The color, in matplotlib notation.
    '''
    ax.yaxis.label.set_color(c)
    ax.xaxis.label.set_color(c)
    ax.tick_params(axis='x', colors=c)
    ax.tick_params(axis='y', colors=c)
    ax.spines['bottom'].set_color(c)
    ax.spines['top'].set_color(c)
    ax.spines['left'].set_color(c)
    ax.spines['right'].set_color(c)
# end change_axes_colors

def make_colorbar(ax, cmap, cnorm, orientation='vertical', ticks=None, ticklabels=None, color='k', alpha=None):
    '''
    Instantiates and returns a colorbar object for the given axes, with a few more options than
    instantiating directly

    Args:
        ax : The axes to make the colorbar on.
        cmap : The Colormap
        norm : The Normalization
        orientation (str, optional) : 'vertical' (default) or 'horizontal' orientation
        ticks (list, optional) : the locations of the ticks. If None will let matplotlib automatically set them.
        ticklabels (list, optional) : the labels of the ticks. If None will let matplotlib automatically set them.
        color (str, optional) : the color of the colorbar, default black.
        alpha (float, optional) : the transparency of the colorbar

    Returns:
        The matplotlib.ColorbarBase object.
    '''
    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=cnorm, orientation=orientation, ticks=ticks, alpha=None)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    if color != 'k':
        matplotlib.rcParams['axes.edgecolor'] = color
        change_axes_colors(ax, color)
    return cb
# end make_colorbar

def argsave():
    """
    A simple argument parser for saving images generated by scripts
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-sf", "--savefile", help="Path to directory to save .png, by default uses local directory figs")
    parser.add_argument("-s", "--save", action="store_true", help="Save the output figure")
    args = parser.parse_args()

    if args.savefile is None:
        svfile = "figs"
    else:
        svfile = args.savefile

    return args.save, svfile
# end argsave

if __name__ == "__init__":
    print("Display code from the QMO Lab")
