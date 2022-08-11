'''
Written by Nathan Neeteson.
A set of basic utilities for importing AIM images into numpy and PyTorch formats.
'''
import vtkbone
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import torch

from bonelab.util.aim_calibration_header import get_aim_density_equation

def get_data_and_pos_from_AIM(fn,convert_to_density=False):
    # The purpose of this function is to get the image data and position from
    # an AIM file. We need the position because the trabecular and cortical
    # masks are trimmed down and we will use the position to line them up
    # with their respective images
    reader = vtkbone.vtkboneAIMReader()
    reader.SetFileName(fn)
    reader.DataOnCellsOff()
    reader.Update()

    image = reader.GetOutput()
    data = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(image.GetDimensions(),order='F')

    if convert_to_density:
        m, b = get_aim_density_equation(reader.GetProcessingLog())
        data = m*data + b

    pos = list(reader.GetPosition())

    return data, pos

def pad_mask_to_image(mask,mask_pos,img):
    return np.pad(
        mask,
        (
            (mask_pos[0],img.shape[0]-mask_pos[0]-mask.shape[0]),
            (mask_pos[1],img.shape[1]-mask_pos[1]-mask.shape[1]),
            (0, 0) # no axial padding needed
        ),
        mode = 'constant'
    )

def format_image_and_masks(img,cort_mask,trab_mask,img_pos,cort_pos,trab_pos,img_ROI_size,mask_ROI_size):

    # figure out where the center of the ROIs should be, which we will simply
    # take to be the center voxel of the cortical mask

    center_y = cort_pos[0] + cort_mask.shape[0]//2
    center_x = cort_pos[1] + cort_mask.shape[1]//2

    # from this we can figure out what should be the minimum and maximum x and
    # y values that we want for the masks

    mask_min_y = center_y - mask_ROI_size//2
    mask_max_y = center_y + mask_ROI_size//2

    mask_min_x = center_x - mask_ROI_size//2
    mask_max_x = center_x + mask_ROI_size//2

    # similarly, we can figure out where the minimum and maximum x and y values
    # should be for the image

    img_min_y = center_y - img_ROI_size//2
    img_max_y = center_y + img_ROI_size//2

    img_min_x = center_x - img_ROI_size//2
    img_max_x = center_x + img_ROI_size//2

    # now we don't know whether we need to trim or pad for any given side of
    # any given dimension to get the masks to the right shape, so the easiest
    # thing to do here is to pad the masks to the size of the image, and then
    # trim them to the ROI

    cort_mask = pad_mask_to_image(cort_mask,cort_pos,img)
    trab_mask = pad_mask_to_image(trab_mask,trab_pos,img)

    img = img[img_min_y:img_max_y,img_min_x:img_max_x,:]
    cort_mask = cort_mask[mask_min_y:mask_max_y,mask_min_x:mask_max_x,:]
    trab_mask = trab_mask[mask_min_y:mask_max_y,mask_min_x:mask_max_x,:]

    # finally, convert the masks to booleans
    cort_mask = np.array(cort_mask>0,dtype=bool)
    trab_mask = np.array(trab_mask>0,dtype=bool)

    return img, cort_mask, trab_mask


def draw_sample(img,cort_mask,trab_mask,back_mask,z,device):

    # ensure we have a numpy array of the type we want
    intensities = np.ascontiguousarray(img,dtype=np.float32)
    # then take out the slice we want, convert it to a torch tensor, and add
    # two empty dimensions to it, one for the batches and one for the channels
    # final dimensions are (b,c,h,w)
    intensities = torch.from_numpy(img[:,:,z]).float().unsqueeze(0).unsqueeze(1).to(device)

    # next, slice out each mask then stack them together into one array
    # the stacking operation coincidentally puts the dimensions in the correct
    # order, (c,h,w)
    labels = np.stack((cort_mask[:,:,z], trab_mask[:,:,z], back_mask[:,:,z]))
    # the stack made a OHE encoding, but the cross entropy loss function wants
    # a single class channel with integer class labels
    # 0 - cortical, 1 - trabecular, 2 - background
    labels = np.argmax(labels, axis=0)
    # finally, convert it to a torch tensor and add an empty dimension for batches
    # final dimensions are (b,c,h,w)
    labels = torch.from_numpy(labels).long().unsqueeze(0).to(device)

    return intensities, labels


def draw_sample_OHE(img,cort_mask,trab_mask,back_mask,z,device):

    # ensure we have a numpy array of the type we want
    intensities = np.ascontiguousarray(img,dtype=np.float32)
    # then take out the slice we want, convert it to a torch tensor, and add
    # two empty dimensions to it, one for the batches and one for the channels
    # final dimensions are (b,c,h,w)
    intensities = torch.from_numpy(img[:,:,z]).float().unsqueeze(0).unsqueeze(1).to(device)

    # next, slice out each mask then stack them together into one array
    # the stacking operation coincidentally puts the dimensions in the correct
    # order, (c,h,w)
    labels = np.stack((cort_mask[:,:,z], trab_mask[:,:,z], back_mask[:,:,z]))
    # the stack made a OHE encoding, but the cross entropy loss function wants
    # a single class channel with integer class labels
    # 0 - cortical, 1 - trabecular, 2 - background
    #labels = np.argmax(labels, axis=0)
    # finally, convert it to a torch tensor and add an empty dimension for batches
    # final dimensions are (b,c,h,w)
    labels = torch.from_numpy(labels).long().unsqueeze(0).to(device)

    return intensities, labels
