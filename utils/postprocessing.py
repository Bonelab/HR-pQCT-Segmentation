"""
Written by Nathan Neeteson
A utility module for post-processing the output predictions from a model that
is estimating the masks for the cortical and trabeculaer compartments of an
HR-pQCT image.
"""
import vtk
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion
from skimage.measure import label as sklabel
from skimage.filters import gaussian, median
from utils.vtk_util import numpy_to_vtkImageData, vtkImageData_to_numpy

import matplotlib.pyplot as plt


def smooth_embedding_field_gaussian(embedding, sigma=2):
    return gaussian(embedding, sigma=sigma, mode='mirror')


def single_mask_plot(ax, m, label):
    ax.imshow(m, cmap='binary')
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title(label)


def debug_mask_plot(m, title, filename=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

    plt.imshow(m[:, :, m.shape[2] // 2], cmap='binary')

    labels = ['y-z', 'x-z', 'x-y']

    for i in range(0, 3):
        single_mask_plot(axs[i], np.swapaxes(m, i, 2)[:, :, m.shape[i] // 2], labels[i])

    fig.suptitle(title)

    if filename:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def convert_mask_to_polydata(mask):
    # construct a poly data surface of the mask with marching cubes
    marching_cubes = vtk.vtkImageMarchingCubes()
    marching_cubes.SetInputData(mask)
    marching_cubes.SetNumberOfContours(1)
    marching_cubes.ComputeScalarsOff()
    marching_cubes.SetValue(0, 0.5)

    marching_cubes.Update()
    return marching_cubes.GetOutput()


def convert_polydata_to_mask(polydata, mask):
    # this requires a filter to translate the polydata to a stencil
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(polydata)
    poly_to_stencil.SetOutputOrigin(mask.GetOrigin())
    poly_to_stencil.SetOutputSpacing(mask.GetSpacing())
    poly_to_stencil.SetOutputWholeExtent(mask.GetExtent())

    poly_to_stencil.Update()

    # and a stencil to write on the image
    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(mask)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)

    stencil.Update()
    return stencil.GetOutput()


def binary_close(mask, n):
    for _ in range(n):
        mask = binary_dilation(mask)

    for _ in range(n):
        mask = binary_erosion(mask)

    return mask


def binary_open(mask, n):
    for _ in range(n):
        mask = binary_erosion(mask)

    for _ in range(n):
        mask = binary_dilation(mask)

    return mask


def keep_largest_connected_component_vtk(mask):
    # convert the mask to vtk image data
    vtk_mask = numpy_to_vtkImageData(mask)

    # convert the image to surface polydata
    polydata = convert_mask_to_polydata(vtk_mask)

    # filter the poly data to keep only the largest connected region
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.ScalarConnectivityOff()
    connectivity.SetExtractionModeToLargestRegion()

    connectivity.Update()
    polydata = connectivity.GetOutput()

    # generate a new vtk image from the filtered poly data
    return vtkImageData_to_numpy(convert_polydata_to_mask(polydata, vtk_mask))


def keep_largest_connected_component_skimage(mask, background=False):
    # use the 'label' function from skimage.measure, aliased as sklabel, to
    # identify separate connected components in a mask and keep only the
    # largest one.
    # mask: numpy binary ndarray of dimension nx,ny,nz
    # background: optional flag, when false filter the mask and when true
    #   filter the background instead

    mask = np.logical_not(mask) if background else mask

    mask = sklabel(mask, background=0)
    mask = mask == np.argmax(np.bincount(mask.flat)[1:]) + 1

    mask = np.logical_not(mask) if background else mask

    return mask


def remove_islands_from_mask(mask, erosion_dilation=None):
    # uses vtk filters to remove the islands from a binary mask, keeping only
    # the largest single connected component

    # pad the mask with zeros to close all surfaces
    mask = np.pad(mask, ((1, 1), (1, 1), (0, 0)), mode='constant')

    if erosion_dilation:
        for _ in range(erosion_dilation):
            mask = binary_erosion(mask)

    mask = keep_largest_connected_component_skimage(mask, background=False)

    if erosion_dilation:
        for _ in range(erosion_dilation):
            mask = binary_dilation(mask)

    # and trim off the padded voxel layers that were added to close surfaces
    mask = mask[1:-1, 1:-1, :]

    return mask


def fill_in_gaps_in_mask(mask, dilation_erosion=None):
    # pad the mask with zeros
    pad_width = 2 * dilation_erosion if dilation_erosion else 1
    mask = np.pad(mask, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant')

    # dilate
    if dilation_erosion:
        for _ in range(dilation_erosion):
            mask = binary_dilation(mask)

    # filter out gaps that remain after dilation
    mask = keep_largest_connected_component_skimage(mask, background=True)

    # erode
    if dilation_erosion:
        for _ in range(dilation_erosion):
            mask = binary_erosion(mask)

    # and trim off the padded voxel layers
    mask = mask[pad_width:-pad_width, pad_width:-pad_width, :]

    # finally, filter out any small gaps that may remain after the dilation and trim
    mask = keep_largest_connected_component_skimage(mask, background=True)

    return mask


def dilate_and_subtract(mask, thickness):
    dilated_mask = mask.copy()

    for _ in range(thickness):
        dilated_mask = binary_dilation(dilated_mask)

    return np.logical_and(dilated_mask, np.logical_not(mask))


def erode_and_subtract(mask, thickness):
    eroded_mask = mask.copy()

    for _ in range(thickness):
        eroded_mask = binary_erosion(eroded_mask)

    return np.logical_and(np.logical_not(eroded_mask), mask)


def extract_bone(image, threshold=-0.25):
    # this function is based on the process for extracting the perisoteal surface
    # described in Buie2007, but is slightly different in that I replaced Buie's
    # dilate>CC>erode sequence with the gap filling function in this file, which
    # is quite similar but a little bit different: pad>dilate>CC>erode>trim>CC
    # I have also added a remove_islands step to get rid of the other
    # bone before proceeding, Buie didn't ned this because they would do some
    # other pre-processing step to get rid of the other bone

    bone_mask = image >= threshold

    bone_mask = median(bone_mask, selem=np.ones((3, 3, 1)))

    bone_mask = remove_islands_from_mask(bone_mask, erosion_dilation=3)

    bone_mask = fill_in_gaps_in_mask(bone_mask, dilation_erosion=15)

    return bone_mask


def iterative_filter(mask, n_islands, n_gaps):
    for N in range(min(n_islands, n_gaps) + 1):
        mask = remove_islands_from_mask(mask, erosion_dilation=N)
        mask = fill_in_gaps_in_mask(mask, dilation_erosion=N)

    if n_islands > n_gaps:
        mask = remove_islands_from_mask(mask, erosion_dilation=n_islands)
    elif n_gaps > n_islands:
        mask = fill_in_gaps_in_mask(mask, dilation_erosion=n_gaps)

    return mask


def postprocess_masks_iterative(
        image,
        cort_mask,
        trab_mask,
        n_islands=5,
        n_gaps=15,
        min_cort_thickness=8,
        morph_bone_threshold=0,
        visualize=False
):
    if visualize:
        debug_mask_plot(trab_mask, '0a. Trabecular mask, input', filename='postproc0a_trab_initial.png')
        debug_mask_plot(cort_mask, '0b. Cortical mask, input', filename='postproc0b_cort_initial.png')

    # Step 1: Iteratively filter the trabecular mask
    trab_mask = iterative_filter(trab_mask, n_islands, n_gaps)

    if visualize:
        debug_mask_plot(trab_mask, '1. Trabecular mask, iteratively filtered', filename='postproc1_trab_filtered.png')

    # Step 2: Generate minimum cortical shell by dilating and subtracting
    #   trabecular mask
    min_cort_mask = dilate_and_subtract(trab_mask, min_cort_thickness)

    if visualize:
        debug_mask_plot(min_cort_mask, '2. Cortical mask, minimum', filename='postproc2_cort_minimum.png')

    # Step 3: Generate a morphological estimate of the bone mask from the image
    morph_bone_mask = extract_bone(image, threshold=morph_bone_threshold)

    if visualize:
        debug_mask_plot(morph_bone_mask, '3. Bone mask, morphological', filename='postproc3_bone_morphological.png')

    # Step 4: Create the bone mask by the union of the trabecular, cortical,
    #   minimum cortical, and morphological bone masks
    bone_mask = np.logical_or(
        np.logical_or(trab_mask, cort_mask),
        np.logical_or(min_cort_mask, morph_bone_mask)
    )
    del min_cort_mask
    del morph_bone_mask

    if visualize:
        debug_mask_plot(bone_mask, '4. Bone mask, composed', filename='postproc4_bone_composed.png')

    # Step 5: Iteratively filter the bone mask
    bone_mask = iterative_filter(bone_mask, n_islands, n_gaps)

    if visualize:
        debug_mask_plot(bone_mask, '5. Bone mask, filtered', filename='postproc5_bone_filtered.png')

    # Step 6: Extract the cortical mask by subtracting the trabecular mask from
    #   the final bone mask
    cort_mask = np.logical_and(bone_mask, np.logical_not(trab_mask))

    if visualize:
        debug_mask_plot(cort_mask, '6. Cort mask, extracted', filename='postproc6_cort_extracted.png')

    return cort_mask, trab_mask
