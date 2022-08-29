import numpy as np
import torch


def convert_image(sample):

    # convert input data to correct type
    image = np.ascontiguousarray(sample['image'], dtype=np.float32)
    # turn it into a torch tensor, and unsqueeze dim=0 to add a dim for
    # the channels (of which we only have one:densities)
    image = torch.from_numpy(image).float().unsqueeze(0)
    # final dim: (c,w,h)

    return image


def convert_labels(sample, ohe):

    # convert the masks to booleans and create back mask
    cort_mask = sample['cort_mask'] > 0
    trab_mask = sample['trab_mask'] > 0
    back_mask = (~cort_mask) & (~trab_mask)

    # stack the masks together
    labels = np.stack((cort_mask, trab_mask, back_mask))
    # what we have created so far is a one hot encoding for the labels. if
    # don't want that, need to take argmax on channel dim
    if not ohe:
        labels = np.expand_dims(np.argmax(labels, axis=0), axis=0)
    # finally, convert to a torch tensor
    labels = torch.from_numpy(labels).long()
    # final dim: (c,w,h)

    return labels


class SampleToTensors(object):
    # the purpose of this transformation object is to take a sample and
    # convert with the images and masks within from numpy arrays to torch
    # tensors with the correct dimensions
    def __init__(self, ohe=True, inference=False):
        self.ohe = ohe
        self.inference = inference

    def __call__(self, sample):

        sample['image'] = convert_image(sample)

        if not self.inference:
            sample['labels'] = convert_labels(sample, self.ohe)

        return sample

