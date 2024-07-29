import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os

from glob import glob
from torch.utils.data import Dataset


class NiftiDataset(Dataset):

    def __init__(
        self,
        data_dir,
        data_pattern,
        transform=None,
        load_masks=True,
        cort_suffix='_cort_mask.nii.gz',
        trab_suffix='_trab_mask.nii.gz'
    ):

        image_file_list = glob(os.path.join(data_dir, data_pattern))

        self.data = {}

        self.data['image'] = [
            image_file for image_file in image_file_list
        ]

        self.data['cort_mask'] = [
            os.path.splitext(image_file)[0] + cort_suffix if load_masks else None
            for image_file in image_file_list
        ]

        self.data['trab_mask'] = [
            os.path.splitext(image_file)[0] + trab_suffix if load_masks else None
            for image_file in image_file_list
        ]

        self.transform = transform

        self.load_masks = load_masks

        self.reader = vtk.vtkNIFTIImageReader()


    def __len__(self):
        return len(self.data["image"])


    def __getitem__(self, idx):

        sample = {}

        self.reader.SetFileName(self.data["image"][idx])
        self.reader.Update()

        sample["image"] = vtk_to_numpy(
            self.reader.GetOutput().GetPointData().GetScalars()
        ).reshape(self.reader.GetOutput().GetDimensions(), order='F')

        sample["image_position"] = [0, 0, 0]

        sample['image_position_original'] = sample['image_position'].copy()
        sample['image_shape_original'] = sample['image'].shape

        self.vtk_image_data = self.reader.GetOutput()

        if self.load_masks:
            for k in ["cort_mask", "trab_mask"]:
                self.reader.SetFileName(self.data[k][idx])
                self.reader.Update()

                sample[k] = vtk_to_numpy(
                    self.reader.GetOutput().GetPointData().GetScalars()
                ).reshape(self.reader.GetOutput().GetDimensions(), order='F')

        else:
            sample["cort_mask"] = np.zeros_like(sample["image"])
            sample["trab_mask"] = np.zeros_like(sample["image"])

        sample['cort_mask_position'] = sample['image_position'].copy()
        sample['trab_mask_position'] = sample['image_position'].copy()

        sample["name"] = os.path.basename(self.data["image"][idx])

        if self.transform:
            sample = self.transform(sample)

        return sample


    def get_vtk_image_data(self):
        return self.vtk_image_data
