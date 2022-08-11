'''
Written by Nathan Neeteson.
Utilities for saving images, for debugging or for saving outputs.
'''

# Imports

import os

import numpy as np
import SimpleITK as sitk
import vtk

from datetime import datetime

from bonelab.util.vtk_util import numpy_to_vtkImageData
from bonelab.io.vtk_helpers import get_vtk_writer, handle_filetype_writing_special_cases

# Functions

def save_numpy_array_as_image(arr,filename):
    image = sitk.GetImageFromArray(arr)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(image)

def save_mask_as_AIM(
        filename,
        mask,
        mask_position,
        image_position,
        image_shape,
        spacing,
        origin,
        processing_log,
        mask_label,
        software,
        version
    ):


    # we never crop/trim images or masks in this workflow, we only pad them.
    # take advantage of that knowledge and realize that we should only ever
    # need to crop the mask back down to the size of the original image, and
    # if this fails then something else has gone wrong

    lower_bounds = np.asarray(image_position) - np.asarray(mask_position)

    mask = mask[
        lower_bounds[0]:(lower_bounds[0]+image_shape[0]),
        lower_bounds[1]:(lower_bounds[1]+image_shape[1]),
        lower_bounds[2]:(lower_bounds[2]+image_shape[2])
    ]



    # append a message to the processing log explaining how and when this
    # mask was created

    processing_log = processing_log + os.linesep + \
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mask_label} created by {software}, version {version}."

    # then, change from a boolean array to an int array where all of the values
    # are 127, convert to vtkImageData, and then write the mask, with the
    # processing log, to an AIM file

    spacing = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
    origin = [float(origin[0]), float(origin[1]), float(origin[2])]

    mask_vtkImageData = numpy_to_vtkImageData(127*mask, spacing=spacing, origin=origin, array_type=vtk.VTK_CHAR)

    writer = get_vtk_writer(filename)
    if writer is None:
        os.sys.exit(f'[ERROR] Cannot find writer for file {filename}')

    writer.SetFileName(filename)

    writer.SetInputData(mask_vtkImageData)

    handle_filetype_writing_special_cases(
        writer,
        processing_log=processing_log
    )

    writer.Update()
