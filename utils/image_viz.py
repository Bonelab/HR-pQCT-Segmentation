'''
Written by Nathan Neeteson.
A set of utilities for visualizing images with masks overlaid in various ways.
'''

import numpy as np
import matplotlib.pyplot as plt

def plot_image_with_mask_overlay(img,cort_mask,trab_mask,back_mask,figsize=(10,6)):

    fig = plt.figure(figsize=figsize)

    mask_min = (img.shape[0]-cort_mask.shape[0])//2
    mask_max = img.shape[0] - (img.shape[0]-cort_mask.shape[0])//2

    plt.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )

    plt.imshow(
        cort_mask,
        cmap='Greens',
        alpha=(0.5*cort_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    plt.imshow(
        trab_mask,
        cmap='Blues',
        alpha=(0.5*trab_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    plt.imshow(
        back_mask,
        cmap='Reds',
        alpha=(0.5*back_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    return fig

def compare_pred_to_labels(slice,predictions,figsize=(12,8)):

    img = slice['image'][0,slice['image'].shape[1]//2,:,:].detach().numpy()
    cort_mask = slice['labels'][0,0,:,:].detach().numpy()
    trab_mask = slice['labels'][0,1,:,:].detach().numpy()
    back_mask = slice['labels'][0,2,:,:].detach().numpy()

    predictions = predictions[0,:,:,:].detach().numpy()

    soft_pred_cort_mask = predictions[0,:,:]
    soft_pred_trab_mask = predictions[1,:,:]
    soft_pred_back_mask = predictions[2,:,:]

    hard_predictions = np.argmax(predictions,axis=0)
    hard_pred_cort_mask = hard_predictions==0
    hard_pred_trab_mask = hard_predictions==1
    hard_pred_back_mask = hard_predictions==2

    soft_cort_error = np.absolute(cort_mask-soft_pred_cort_mask)
    hard_cort_error = np.absolute(cort_mask-hard_pred_cort_mask)

    mask_min = (img.shape[0]-cort_mask.shape[0])//2
    mask_max = img.shape[0] - (img.shape[0]-cort_mask.shape[0])//2

    fig, ((ax11, ax12, ax13), (ax21, ax22, ax23)) = \
        plt.subplots(2,3,sharex=True, sharey=True)

    # plot original segmentation
    ax11.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )
    ax11.imshow(
        cort_mask,
        cmap='Greens',
        alpha=(0.5*cort_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )
    ax11.imshow(
        trab_mask,
        cmap='Blues',
        alpha=(0.5*trab_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )
    ax11.imshow(
        back_mask,
        cmap='Reds',
        alpha=(0.5*back_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    # plot predicted segmentation with softmax
    ax12.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )
    ax12.imshow(
        soft_pred_cort_mask,
        cmap='Greens',
        alpha=(0.5*soft_pred_cort_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )
    ax12.imshow(
        soft_pred_trab_mask,
        cmap='Blues',
        alpha=(0.5*soft_pred_trab_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )
    ax12.imshow(
        soft_pred_back_mask,
        cmap='Reds',
        alpha=(0.5*soft_pred_back_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    # plot predicted segmentation with hardmax
    ax13.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )
    ax13.imshow(
        hard_pred_cort_mask,
        cmap='Greens',
        alpha=(0.5*hard_pred_cort_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )
    ax13.imshow(
        hard_pred_trab_mask,
        cmap='Blues',
        alpha=(0.5*hard_pred_trab_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )
    ax13.imshow(
        hard_pred_back_mask,
        cmap='Reds',
        alpha=(0.5*hard_pred_back_mask),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    # plot just the image, for reference
    ax21.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )

    # plot errors in cort segmentation with softmax
    ax22.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )
    ax22.imshow(
        soft_cort_error,
        cmap='Reds',
        alpha=(0.5*soft_cort_error),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    # plot errors in cort segmentation with hardmax
    ax23.imshow(
        img,
        cmap='gray',
        extent=[0,img.shape[0],0,img.shape[1]]
    )
    ax23.imshow(
        hard_cort_error,
        cmap='Reds',
        alpha=(0.5*hard_cort_error),
        extent=[mask_min, mask_max, mask_min, mask_max]
    )

    plt.subplots_adjust(
        left =      0.05,   right =     0.95,
        bottom =    0.05,   top =       0.95,
        wspace =    0.05,   hspace =    0.05
    )

    return fig

def single_axis_image_and_masks(ax,image,masks,colors,title=None):
    # ax: the plot axis to plot things on
    # image: a 2D gray-scale image as an array
    # masks: a list of binary masks as 2D arrays
    # colors: a list of colormap colors corresponding to masks
    # title: a string, to put over the plot

    ax.imshow(
        image,
        cmap = 'gray'
    )

    for mask,color in zip(masks,colors):

        ax.imshow(
            mask,
            cmap=color,
            alpha=(0.5*mask)
        )

    if title:
        ax.set_title(title)

    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
