from torch.utils.data import Dataset


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
            slice_idx.insert(0, max(idx - (i + 1), 0))
            slice_idx.append(min(idx + (1 + 1), self.image.shape[-1] - 1))

        # finally, construct the sample dict
        sample = {}
        sample['image'] = self.image[:, :, :, slice_idx]
        # if we have more than one slice, we want to switch the dim where the
        # slices are to the channel dim location, in this case the first dim
        if len(slice_idx) > 1:
            sample['image'] = sample['image'].transpose(0, 3)
        # then we squeeze off the last dim, which will be of length 1 now
        # whether we had multiple slices
        sample['image'] = sample['image'].squeeze(3)
        sample['labels'] = self.labels[:, :, :, idx]

        # finally apply whatever transforms were given to the dataset
        if self.transform:
            sample = self.transform(sample)

        return sample

    # these methods allow new images to be slotted into this dataset

    def _set_image(self, image):
        self.image = image[0, :, :, :, :]

    def _set_labels(self, labels):
        self.labels = labels[0, :, :, :, :]

    def setup_new_image(self, sample):
        self._set_image(sample['image'])
        self._set_labels(sample['labels'])
