'''
Written by Nathan Neeteson.
A set of functions for plotting the results of model outputs on a test set.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# this function just adds an annotation to a plot that shows the slope, intercept,
# and rsquared values (with 95% CI) on a linear fit plot
def linear_fit_annotation(ax,linfit,minval,maxval):
    ax.annotate(f'''Slope: {linfit.slope:.2f}$\pm${(1.96*linfit.stderr):.2f}
Intercept: {np.abs(linfit.intercept):.2f}$\pm${(1.96*linfit.intercept_stderr):.2f}
R$^2$ = {linfit.rvalue:.2f}''',
        ((maxval+2*minval)/3,minval),
        fontsize='small'
    )

# simple numpy style function for calculating the single-class dice coefficient
def dice_coefficient(X,Y):
    # X, Y: binary masks of the same size
    X, Y = X.flatten(), Y.flatten()
    return 2*np.sum(X*Y) / (np.sum(X) + np.sum(Y))


def linear_correlation_plot(ax,data1,data2,label1,label2,parameter,units,show_title=True,*args,**kwargs):
    # ax: axes object to plot on
    # data1, data2: same-size lists of values of same parameter from two sets
    # label1, label2: strings
    # parameter, units: strings

    # convert data to numpy arrays
    data1, data2 = np.asarray(data1), np.asarray(data2)

    # find the smallest and largest values in all the data
    minval = np.min([data1.flatten(),data2.flatten()])
    maxval = np.max([data1.flatten(),data2.flatten()])

    # plot the data against each other
    ax.scatter(data1,data2,*args,**kwargs)

    # plot a slope=1,int=0 line covering the range of data
    ax.plot([minval,maxval],[minval,maxval],color='black',linestyle='--')

    # fit a linear model to the datasets
    linfit = linregress(data1,data2)

    # plot the fitted line
    ax.plot(
        np.unique(data1), linfit.slope*np.unique(data1)+linfit.intercept,
        color = 'red', linestyle = '--'
    )

    # annotate the linear fit on the plot
    #linear_fit_annotation(ax,linfit,minval,maxval)

    # limits and tick marks
    #ax.xlim([minval,maxval])
    #ax.ylim([minval,maxval])

    # axis labels
    ax.set_xlabel(fr'{label1} [{units}]')
    ax.set_ylabel(fr'{label2} [{units}]')

    ax.grid()

    if show_title:
        ax.set_title(f'{parameter}')

def bland_altman_plot(ax,data1,data2,parameter,units,show_title=True,ylim=None,*args,**kwargs):
    # adapted from answer given here:
    # https://stackoverflow.com/questions/16399279/bland-altman-plot-in-python
    # ax: axes object to plot on
    # data1, data2: same-size lists of values of same parameter from two sets
    # parameter, units: strings

    # convert the data to numpy arrays
    data1, data2 = np.asarray(data1), np.asarray(data2)

    # calculate the mean and difference of each set of datapoints
    mean = (data1 + data2) / 2
    diff = data2 - data1

    # set the y limits if they are not set
    if not(ylim):
        mean_mean = np.mean(np.abs(mean))
        max_diff = 1.05*np.max(np.abs(diff))
        ylim_val = max(0.1*mean_mean,max_diff)
        ylim = [-ylim_val,ylim_val]

    # find the mean and std of the differences
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    ax.grid()

    # plot the differences against the means
    ax.scatter(mean,diff,*args,**kwargs)

    # add a line to emphasize where 0 difference is
    ax.axhline(0,color='black',linestyle='-')

    # fit a linear model to the mean vs diff
    linfit = linregress(mean,diff)

    '''
    # plot the fitted line to see if there is a trend in the error
    ax.plot(
        np.unique(mean), linfit.slope*np.unique(mean)+linfit.intercept,
        color = 'red', linestyle = '--'
    )
    '''

    # plot the mean of the errors
    ax.axhline(diff_mean,color='red',linestyle='--')

    # show the confidence interval in the differences
    ax.axhline(diff_mean+1.96*diff_std,color='gray',linestyle='--')
    ax.axhline(diff_mean-1.96*diff_std,color='gray',linestyle='--')

    # axis labels
    ax.set_xlabel(fr'Mean [{units}]')
    ax.set_ylabel(fr'Predicted - Reference [{units}]')

    if ylim:
        ax.set_ylim(ylim[0],ylim[1])

    if show_title:
        ax.set_title(f'{parameter}')

    ax.text(
        0.95, 0.95,
        fr'Bias: {diff_mean:0.3f} (SE {(1.96*diff_std):0.3f}) {units}',
        horizontalalignment='right',
        verticalalignment='top',
        transform = ax.transAxes,
        bbox=dict(facecolor='white', edgecolor='black', pad=5.0)
    )

def relative_error_plot(ax,data1,data2,parameter,units,show_title=True,ylim=None,*args,**kwargs):

    EPS = 1e-8

    # convert the data to numpy arrays
    data1, data2 = np.asarray(data1), np.asarray(data2)

    relative_error = 100*(data2 - data1) / (data1+EPS)

    relative_error_mean = np.mean(relative_error)
    relative_error_std = np.std(relative_error)

    ax.grid()

    ax.scatter(data1,relative_error,*args,**kwargs)

    ax.axhline(0,color='black',linestyle='-')

    # plot the mean of the errors
    ax.axhline(relative_error_mean,color='red',linestyle='--')

    # show the confidence interval in the differences
    ax.axhline(relative_error_mean+1.96*relative_error_std,color='gray',linestyle='--')
    ax.axhline(relative_error_mean-1.96*relative_error_std,color='gray',linestyle='--')

    ax.set_xlabel(fr'Reference [{units}]')
    ax.set_ylabel('Relative Error [%]')

    if ylim:
        ax.set_ylim(ylim[0],ylim[1])

    if show_title:
        ax.set_title(f'{parameter}')

    ax.text(
        0.95, 0.95,
        fr'{relative_error_mean:0.1f}% $\pm$ {(1.96*relative_error_std):0.1f}%',
        horizontalalignment='right',
        verticalalignment='top',
        transform = ax.transAxes
    )


def error_historgram_plot(ax,data1,data2,parameter,units,show_title=True,show_ylabel=True,ylim=None,*args,**kwargs):
    # ax: axes object to plot on
    # data1, data2: same-size lists of values of same parameter from two sets
    # parameter, units: strings

    # calculate the error assuming data1 is the reference
    err = data2 - data1

    # plot an error historgram horizontally
    ax.hist(err,bins='sqrt',rwidth=0.8,orientation='horizontal',*args,**kwargs)

    # axis labels
    ax.set_xlabel(f'Counts')
    if show_ylabel:
        ax.set_ylabel(f'Error [{units}]')

    ax.yaxis.grid(True)

    if ylim:
        ax.set_ylim(ylim[0],ylim[1])

    if show_title:
        ax.set_title(f'{parameter}')

def comparison_triplet(axs,data1,data2,label1,label2,parameter,units,show_title=True,ylim_margin_factor=0.15):
    # axs: list of 3 axes objects to plot on
    # data1, data2: same-size lists of values of same parameter from two sets
    # label1, label2: strings
    # parameter, units: strings

    data1, data2 = np.asarray(data1), np.asarray(data2)

    # compute the min and maximum difference between the datasets
    mindiff = np.min((data2-data1).flatten())
    maxdiff = np.max((data2-data1).flatten())

    # then set the min and max y values for the bland altman and error historgram
    # plots to include all of the values plus some extra margin, so that the
    # y axis limits on these plots line up
    mindiff, maxdiff = mindiff-ylim_margin_factor*(maxdiff-mindiff),maxdiff+ylim_margin_factor*(maxdiff-mindiff)

    diff_ylim = [mindiff,maxdiff]

    linear_correlation_plot(
        axs[0], data1, data2, label1, label2, parameter, units, show_title=False,
        edgecolors='black', facecolors='none'
    )

    bland_altman_plot(
        axs[1],data1,data2,parameter,units,show_title=show_title,ylim=diff_ylim,
        edgecolors='black', facecolors='none'
    )

    error_historgram_plot(
        axs[2],data1,data2,parameter,units,show_title=False,show_ylabel=False,ylim=diff_ylim,
        color='black'
    )

def multi_violin_plot(ax,data,labels,parameter,units,show_ylabel=True,*args,**kwargs):
    # ax: axes object to plot on
    # data: list of data series
    # label: list of strings corresponding to data series
    # parameter, units: strings

    data = np.asarray(data).T

    ax.violinplot(data,*args,**kwargs)

    ax.yaxis.grid(True)
    plt.setp(ax,
        xticks = [y + 1 for y in range(data.shape[1])],
        xticklabels = labels
    )

    if show_ylabel:
        ax.set_ylabel(f'{parameter} [{units}]')
