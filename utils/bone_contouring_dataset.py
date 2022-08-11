'''
Written by Nathan Neeteson.
A set of classes for controlling the flow of data (datasets) and for
pre-processing inputs to the model (padding, normalization, type conversion).
'''

from bonelab.util.aim_calibration_header import get_aim_density_equation
import vtkbone
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
import math

import os
import glob

# NOTE: Consider replacing these class defs for the transformations with
# factory functions instead. Would make the code cleaner, but unsure if
# it would be compatible with pytorch's transforms.Compose()


# NOTE: In the end I replaced usage of the sample cropping class with the
# sample padding class, and converted the model to use same-style convolutions.
# This remains here even though there is some sort of problem somewhere with the
# logic that tries to keep model outputs lined up with the original image.
# User beware! For BMEN619 please ignore this class since it is gigantic, not
# used, and apparently has some kind of bug.
class SampleCropper(object):
    # because of how the UNet works, the image and the masks have to be cropped
    # to different sizes, but in a way that ensures they will be properly aligned.
    # this class allows us to create a cropping "function" that will crop the
    # images in a way that works for the network we're currently training
    def __init__(self,model,device,n_channels=1,with_embeddings=False,min_img_size=300,max_img_size=1000,inc_img_size=50,pad_mode='edge'):
        # to create the cropper, we hand it the model instance so it can figure
        # out for itself what are the (image_size,mask_size) pairs

        self.image_sizes = []
        self.mask_sizes = []
        self.n_channels = n_channels
        self.pad_mode = pad_mode

        if with_embeddings:
            self.seg_types = ['mask', 'embedding']
            self.pad_val = {'mask': 0, 'embedding': 0.5}
        else:
            self.seg_types = ['mask']
            self.pad_val = {'mask': 0}

        self.regions = ['cort', 'trab']

        print('Building sample cropper...')

        # iterate over the allowable image sizes, creating tensors of the
        # image size and sending them through and observing the output size
        for img_size in range(min_img_size,max_img_size+1,inc_img_size):
            print(f'Checking img_size={img_size}. Increment:{inc_img_size}, Maximum:{max_img_size}.')
            # we put this inside a try block because it's possible that a given
            # input image size is too small for it to make it all the way
            # to the bottom of the UNet, and we don't want to have the script
            # stop due to an error, just want to move on to next candidate size
            try:
                input = np.zeros((1,self.n_channels,img_size,img_size),dtype=np.float32)
                input = torch.from_numpy(input).float()
                output = model(input)
                self.image_sizes.append(input.shape[3])
                self.mask_sizes.append(output.shape[3])
            except Exception as e:
                print(e)

    def __call__(self,sample):
        # when this object is invoked on a training sample, it will crop the
        # image and the masks to sizes that are compatible with the model.
        # the procedure is:
        # 1. find the largest extent of the masks
        max_mask_extent = max(
            sample['cort_mask'].shape[0], sample['cort_mask'].shape[1],
            sample['trab_mask'].shape[0], sample['trab_mask'].shape[1]
        )

        # 2. select the next largest mask size, also setting the image size
        idx_size = self.select_size_index(max_mask_extent)
        image_size = self.image_sizes[idx_size]
        mask_size = self.mask_sizes[idx_size]

        # 3. pad or crop the image so it is the right size
        # we have to also modify the position value for the image so that the
        # masks get correctly aligned
        if sample['image'].shape[0] < image_size:
            # pad the y dimension
            diff = image_size - sample['image'].shape[0]
            p1, p2 = diff//2, diff//2
            p2 += 1 if (diff%2)==1 else 0
            sample['image'] = np.pad(sample['image'],((p1,p2),(0,0),(0,0)),mode=self.pad_mode)
            sample['image_position'][0] -= p1

        elif sample['image'].shape[0] > image_size:
            # trim the y dimension
            diff = sample['image'].shape[0] - image_size
            t1, t2 = diff//2, diff//2
            t2 += 1 if (diff%2)==1 else 0
            sample['image'] = sample['image'][t1:-t2,:,:]
            sample['image_position'][0] += t1

        if sample['image'].shape[1] < image_size:
            # pad the x dimension
            diff = image_size - sample['image'].shape[1]
            p1, p2 = diff//2, diff//2
            p2 += 1 if (diff%2)==1 else 0
            sample['image'] = np.pad(sample['image'],((0,0),(p1,p2),(0,0)),mode=self.pad_mode)
            sample['image_position'][1] -= p1

        elif sample['image'].shape[1] > image_size:
            # trim the x dimension
            diff = sample['image'].shape[1] - image_size
            t1, t2 = diff//2, diff//2
            t2 += 1 if (diff%2)==1 else 0
            sample['image'] = sample['image'][:,t1:-t2,:]
            sample['image_position'][0] += t1

        # 4. pad the masks to the size of the image so they are aligned

        for r in self.regions:
            for s in self.seg_types:
                sample[f'{r}_{s}'] = self.pad_mask_to_image(
                    sample['image'], sample['image_position'],
                    sample[f'{r}_{s}'], sample[f'{r}_mask_position'],
                    self.pad_val[s]
                )
            sample[f'{r}_mask_position'] = sample['image_position']

        # 6. trim the mask to the approriate size
        # at this point we should not have to worry about odd differences,
        # because convolution and downsampling should remove voxels symmetrically

        t = (image_size - mask_size)//2
        for r in self.regions:
            for s in self.seg_types:
                sample[f'{r}_{s}'] = sample[f'{r}_{s}'][t:-t,t:-t,:]
            sample[f'{r}_mask_position'][0] += t
            sample[f'{r}_mask_position'][1] += t

        # finally return the sample dictionary
        return sample

    def select_size_index(self,max_mask_extent):
        mask_sizes = np.asarray(self.mask_sizes)
        mask_sizes[mask_sizes<max_mask_extent] = -2*max(mask_sizes)
        return (np.abs(mask_sizes-max_mask_extent)).argmin()

    def pad_mask_to_image(self,img,img_pos,mask,mask_pos,pad_val):
        return np.pad(
            mask,
            (
                (mask_pos[0]-img_pos[0],(img_pos[0]+img.shape[0])-(mask_pos[0]+mask.shape[0])),
                (mask_pos[1]-img_pos[1],(img_pos[1]+img.shape[1])-(mask_pos[1]+mask.shape[1])),
                (0, 0) # no axial padding needed
            ),
            mode = 'constant',
            constant_values = pad_val
        )

class SamplePadder(object):
    # this transformation object will pad the image and the masks so that:
    # - they are the same size
    # - they are properly aligned
    # - they are square
    # - the side length is a multiple of <img_size_mult>
    # optionally, the transformation can augment the image sample by randomly
    # padding more or less on either side of the x and y dimensions, this is
    # a good idea so that the model is not biased based on how well-centered
    # the training samples are
    def __init__(self,img_size_mult,pad_mode='edge',augmentation=False):
        self.img_size_mult = img_size_mult
        self.pad_mode = pad_mode
        self.augmentation = augmentation

    def __call__(self,sample):
        # when a sample is passed through this transformation, the output
        # should be of images and masks that are all of the same size, aligned,
        # and who have x and y dim lengths the same and a multiple of the integer
        # provided at object initialization.

        volumes = ['image','cort_mask','trab_mask']
        volumes = list(set(volumes).intersection(sample.keys()))

        # Step 1: Figure out the upper and lower x,y bounds for the union of
        # the image and masks

        lower_x = min([sample[f"{v}_position"][0] for v in volumes])
        upper_x = max([sample[f"{v}_position"][0]+sample[v].shape[0] for v in volumes])

        lower_y = min([sample[f"{v}_position"][1] for v in volumes])
        upper_y = max([sample[f"{v}_position"][1]+sample[v].shape[1] for v in volumes])

        # Step 2: Calculate the new size for the image, which will be the larger
        # of the width and height, then rounded up to the next highest multiple

        width = upper_x - lower_x
        height = upper_y - lower_y

        padded_size = self.img_size_mult*math.ceil(max(width,height)/self.img_size_mult)

        # Step 3: Calculate the new upper/lower bounds in x and y

        lower_x_padded = lower_x - (padded_size-width)//2
        upper_x_padded = lower_x_padded + padded_size

        lower_y_padded = lower_y - (padded_size-height)//2
        upper_y_padded = lower_y_padded + padded_size

        # Step 4: Pad the image and masks to the new bounds, and adjust the
        # position entries in the sample dict

        for v in volumes:


            if v=='image':
                pad_mode = self.pad_mode
            else:
                pad_mode = 'constant'

            sample[v] = np.pad(
                sample[v],
                (
                    (
                        sample[f"{v}_position"][0] - lower_x_padded,
                        upper_x_padded - (sample[f"{v}_position"][0] + sample[v].shape[0])
                    ),
                    (
                        sample[f"{v}_position"][1] - lower_y_padded,
                        upper_y_padded - (sample[f"{v}_position"][1] + sample[v].shape[1])
                    ),
                    (0, 0)
                ),
                mode = pad_mode
            )

            sample[f'{v}_position'][0] = lower_x_padded
            sample[f'{v}_position'][1] = lower_y_padded

        # Step 5: If augmentation is enabled, then randomly pad the image
        # to the next size multiple, to de-center the actual bone

        if self.augmentation:

            # we will randomly choose whether to pad the left/right top/bottom
            # in x and y, using numpy's random binomial function to flip a coin
            # independently for each

            if np.random.binomial(1,0.5):
                aug_pad_x_lower = self.img_size_mult
                aug_pad_x_upper = 0
            else:
                aug_pad_x_lower = 0
                aug_pad_x_upper = self.img_size_mult

            if np.random.binomial(1,0.5):
                aug_pad_y_lower = self.img_size_mult
                aug_pad_y_upper = 0
            else:
                aug_pad_y_lower = 0
                aug_pad_y_upper = self.img_size_mult

            # then we loop through the volumes and apply the augmentation pad

            for v in volumes:

                if v=='image':
                    pad_mode = self.pad_mode
                else:
                    pad_mode = 'constant'

                sample[v] = np.pad(
                    sample[v],
                    (
                        (aug_pad_x_lower, aug_pad_x_upper),
                        (aug_pad_y_lower, aug_pad_y_upper),
                        (0, 0)
                    ),
                    mode = pad_mode
                )

                sample[f'{v}_position'][0] = sample[f'{v}_position'][0] - aug_pad_x_lower
                sample[f'{v}_position'][1] = sample[f'{v}_position'][1] - aug_pad_y_lower

        return sample


class SampleStandardizer(object):
    # this transformation object is very simple. At creation, a mean and
    # standard deviation are specified. When a sample is passed through, the
    # image portion of the sample is normalized so that the intensity values
    # are capped to the given range, then scaled to the interval [-1,1]
    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self,sample):

        sample['image'] = np.minimum(np.maximum(sample['image'],self.min_val),self.max_val)
        sample['image'] = (2*sample['image'] - self.max_val - self.min_val) / (self.max_val - self.min_val)

        return sample

class SampleToTensors(object):
    # the purpose of this transformation object is to take a sample and
    # convert with the images and masks within from numpy arrays to torch
    # tensors with the correct dimensions
    def __init__(self,ohe=True,inference=False):
        # at creation, we tell the object what device we are using (cuda/cpu)
        # and whether or not we want the labels as a one-hot encoding
        self.ohe = ohe
        self.inference = inference

    def __call__(self,sample):

        sample['image'] = self.convert_image(sample)

        if not(self.inference):
            sample['labels'] = self.convert_labels(sample)

        return sample

    def convert_image(self,sample):

        # convert input data to correct type
        image = np.ascontiguousarray(sample['image'],dtype=np.float32)
        # turn it into a torch tensor, and unsqueeze dim=0 to add a dim for
        # the channels (of which we only have one:densities)
        image = torch.from_numpy(image).float().unsqueeze(0)
        # final dim: (c,w,h)

        return image

    def convert_labels(self,sample):

        # convert the masks to booleans and create back mask
        cort_mask = sample['cort_mask']>0
        trab_mask = sample['trab_mask']>0
        back_mask = (~cort_mask)&(~trab_mask)

        # stack the masks together
        labels = np.stack((cort_mask,trab_mask,back_mask))
        # what we have created so far is a one hot encoding for the labels. if
        # don't want that, need to take argmax on channel dim
        if not(self.ohe):
            labels = np.expand_dims(np.argmax(labels,axis=0), axis=0)
        # finally, convert to a torch tensor
        labels = torch.from_numpy(labels).long()
        # final dim: (c,w,h)

        return labels


class BoneContouringDataset(Dataset):
    # this dataset will keep track of all of the data in a directory, and
    # load the data and output on demand

    def __init__(self, data_dir, transform=None):
        # data_dir should point to a folder containing npz files as created
        # by the preprocess_vtk_images function in the preprocessing module.

        self.data_list = glob.glob(f'{data_dir}*.npz')
        self.data_list.sort()

        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # when an item is requested, load from file and convert it to a dict
        sample = dict(np.load(self.data_list[idx]))

        # then apply whatever transforms were given to this dataset
        if self.transform:
            sample = self.transform(sample)

        return sample

class HRpQCT_AIM_Dataset(Dataset):

    def __init__(self, data_dir, transform=None, load_masks=True):

        # the image files should have the pattern <study>_<ID#>_<site_code>.AIM
        image_file_list = glob.glob(f'{data_dir}*_*_??.AIM')

        self.data = {
            'image': [],
            'cort_mask': [],
            'trab_mask': []
        }

        # each image in the directory also needs to have both a cort and trab
        # mask, so get those right now
        for image_file in image_file_list:

            self.data['image'].append(image_file)

            image_file_noext = os.path.splitext(image_file)[0]

            if load_masks:

                cort_mask_file = image_file_noext+'_CORT_MASK.AIM'

                if os.path.exists(cort_mask_file):
                    self.data['cort_mask'].append(cort_mask_file)
                else:
                    raise Exception(f'Expected file, {cort_mask_file}, does not exist.')


                trab_mask_file = image_file_noext+'_TRAB_MASK.AIM'

                if os.path.exists(trab_mask_file):
                    self.data['trab_mask'].append(trab_mask_file)
                else:
                    raise Exception(f'Expected file, {trab_mask_file}, does not exist.')

            else:
                self.data['cort_mask'].append(None)
                self.data['trab_mask'].append(None)

        # store the given transform(s)
        self.transform = transform

        # store if loading masks
        self.load_masks = load_masks

        # create the AIM Reader
        self.reader = vtkbone.vtkboneAIMReader()
        self.reader.DataOnCellsOff()

    def __len__(self):
        return len(self.data['image'])

    def __getitem__(self,idx):

        sample = {}

        sample['image'], sample['image_position'] = \
            self._get_image_data_and_position(self.data['image'][idx],True)

        sample['spacing'], sample['origin'] = self._get_spacing_and_origin(self.data['image'][idx])
        sample['processing_log'] = self._get_processing_log(self.data['image'][idx])

        sample['image_position_original'] = sample['image_position'].copy()
        sample['image_shape_original'] = sample['image'].shape

        if self.load_masks:
            for k in ['cort_mask', 'trab_mask']:
                sample[k], sample[f'{k}_position'] = \
                    self._get_image_data_and_position(self.data[k][idx],False)
        else:
            sample['cort_mask'] = np.zeros_like(sample['image'])
            sample['trab_mask'] = np.zeros_like(sample['image'])

            sample['cort_mask_position'] = sample['image_position'].copy()
            sample['trab_mask_position'] = sample['image_position'].copy()

        sample['name'] = os.path.basename(self.data['image'][idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_image_data_and_position(self,filename,convert_to_density=False):

        self.reader.SetFileName(filename)
        self.reader.Update()

        data = vtk_to_numpy(
            self.reader.GetOutput().GetPointData().GetScalars()
        ).reshape(self.reader.GetOutput().GetDimensions(),order='F')

        if convert_to_density:
            m, b = get_aim_density_equation(self.reader.GetProcessingLog())
            data = m*data + b

        position = list(self.reader.GetPosition())

        return data, position

    def _get_processing_log(self,filename):

        self.reader.SetFileName(filename)
        self.reader.Update()

        return self.reader.GetProcessingLog()

    def _get_spacing_and_origin(self,filename):

        self.reader.SetFileName(filename)
        self.reader.Update()

        return self.reader.GetOutput().GetSpacing(), self.reader.GetOutput().GetOrigin()



# TODO: Clean this up so there is one singleimagedataset class with the shared
# featyres, and two classes that inherit from it and then implement methods for
# providing either stacks of 2D slices, or 3D tiles

class SingleImageDataset(Dataset):
    # the purpose of this class is to contain a single image (and labels) and
    # let us use a dataloader to get batches of slices. the optional parameter
    # num_adjacent_slices controls how many adjacent slices to grab as inputs
    def __init__(self, transform=None, num_adjacent_slices=0):
        self.transform = transform
        self.num_adjacent_slices = num_adjacent_slices

    def __len__(self):
        return self.image.shape[-1]

    def __getitem__(self, idx):

        # first, create a list with just the current slice
        slice_idx = [idx]

        # then for each adjacent slice we want, we add another slice to the
        # front and back of the list, with some logic to ensure we do not
        # exceed the bounds of the image
        for i in range(self.num_adjacent_slices):
            slice_idx.insert(0,max(idx-(i+1),0))
            slice_idx.append(min(idx+(1+1),self.image.shape[-1]-1))

        # finally, construct the sample dict
        sample = {}
        sample['image'] = self.image[:,:,:,slice_idx]
        # if we have more than one slice, we want to switch the dim where the
        # slices are to the channel dim location, in this case the first dim
        if len(slice_idx)>1:
            sample['image'] = sample['image'].transpose(0,3)
        # then we squeeze off the last dim, which will be of length 1 now
        # whether or not we had multiple slices
        sample['image'] = sample['image'].squeeze(3)
        sample['labels'] = self.labels[:,:,:,idx]

        # finally apply whatever transforms were given to the dataset
        if self.transform:
            sample = self.transform(sample)

        return sample

    # these methods allow new images to be slotted into this dataset

    def _set_image(self, image):
        self.image = image[0,:,:,:,:]

    def _set_labels(self, labels):
        self.labels = labels[0,:,:,:,:]

    def setup_new_image(self,sample):
        self._set_image(sample['image'])
        self._set_labels(sample['labels'])


class SingleImageTilingDataset(Dataset):

    def __init__(self, tile_shape, transform=None):
        self.tile_shape = np.asarray(tile_shape)
        self.transform = transform

        self.num_tiles = 0

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):

        # construct the sample dict
        sample = {}

        # get the needed indexing bounds for slicing out the tile:
        ((xl,xu),(yl,yu),(zl,zu)) = self._get_tile_bounds(idx)

        # get the image and labels
        sample['image'] = self.image[:,xl:xu,yl:yu,zl:zu]
        sample['labels'] = self.labels[:,xl:xu,yl:yu,zl:zu]

        if not(np.all(sample['image'].shape)):
            print(f'image shape: {self.image.shape}')
            print(f'tile shape: {self.tile_shape}')
            print(f'x;:{xl},xu:{xu},yl:{yl},yu:{yu},zl:{zl},zu"{zu}')

        # apply transformations if applicable
        if self.transform:
            sample = self.transform(sample)

        return sample

    def _set_image(self, image):
        self.image = image[0,:,:,:,:]

    def _set_labels(self, labels):
        self.labels = labels[0,:,:,:,:]

    def _init_preds(self):
        self.preds = torch.zeros_like(self.labels)

    def _create_tiles(self):
        # first, figure out how many tiles needed in each dimension
        n = np.ceil(self.image.shape[1:4] / self.tile_shape).astype('int')
        self.num_tiles = np.prod(n)
        # next, for each dimension (x,y,z) construct a list of lower indices
        # for each tile, making sure that the final tile doesn't go past the
        # edge of the image
        lower = []
        for i in range(3):
            #lower.append(self.tile_shape[i]*np.arange(0,n[i]))
            #lower[-1][-1] = self.image.shape[i+1] - self.tile_shape[i]
            lower.append(np.linspace(0,self.image.shape[i+1]-self.tile_shape[i],n[i]).round().astype(int))

        # then construct the lists of lower indices in each dimension for all
        # of the tiles
        x, y, z = np.meshgrid(*lower)
        self.tile_idxs = np.asarray([x.flatten(),y.flatten(),z.flatten()])

    def _get_tile_bounds(self,idx):
        xl, yl, zl = self.tile_idxs[0][idx], self.tile_idxs[1][idx], self.tile_idxs[2][idx]
        xu, yu, zu = xl+self.tile_shape[0], yl+self.tile_shape[1], zl+self.tile_shape[2]

        return ((xl,xu),(yl,yu),(zl,zu))

    def setup_new_sample(self,image,labels):
        self._set_image(image)
        self._set_labels(labels)
        self._create_tiles()
        self._init_preds()

    def set_tile_preds(self,idx,preds):

        # get the needed indexing bounds for slicing out the tile:
        ((xl,xu),(yl,yu),(zl,zu)) = self._get_tile_bounds(idx)

        # assign values to the relevant tile of the overall pred tensor
        self.preds[:,xl:xu,yl:yu,zl:zu] = torch.argmax(preds,dim=1)

    def get_image_preds(self,preds):
        return self.preds
