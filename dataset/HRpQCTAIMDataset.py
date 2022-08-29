from bonelab.util.aim_calibration_header import get_aim_density_equation
import vtkbone
import os
from vtk.util.numpy_support import vtk_to_numpy

from glob import glob
from torch.utils.data import Dataset


class HRpQCTAIMDataset(Dataset):

    def __init__(self, data_dir, transform=None, load_masks=True):

        # the image files should have the pattern <study>_<ID#>_<site_code>.AIM
        image_file_list = glob(os.path.join(data_dir, '*_*_??.AIM'))

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

                cort_mask_file = image_file_noext + '_CORT_MASK.AIM'

                if os.path.exists(cort_mask_file):
                    self.data['cort_mask'].append(cort_mask_file)
                else:
                    raise Exception(f'Expected file, {cort_mask_file}, does not exist.')

                trab_mask_file = image_file_noext + '_TRAB_MASK.AIM'

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

    def __getitem__(self, idx):

        sample = {}

        sample['image'], sample['image_position'] = \
            self._get_image_data_and_position(self.data['image'][idx], True)

        sample['spacing'], sample['origin'] = self._get_spacing_and_origin(self.data['image'][idx])
        sample['processing_log'] = self._get_processing_log(self.data['image'][idx])

        sample['image_position_original'] = sample['image_position'].copy()
        sample['image_shape_original'] = sample['image'].shape

        if self.load_masks:
            for k in ['cort_mask', 'trab_mask']:
                sample[k], sample[f'{k}_position'] = \
                    self._get_image_data_and_position(self.data[k][idx], False)
        else:
            sample['cort_mask'] = np.zeros_like(sample['image'])
            sample['trab_mask'] = np.zeros_like(sample['image'])

            sample['cort_mask_position'] = sample['image_position'].copy()
            sample['trab_mask_position'] = sample['image_position'].copy()

        sample['name'] = os.path.basename(self.data['image'][idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _get_image_data_and_position(self, filename, convert_to_density=False):

        self.reader.SetFileName(filename)
        self.reader.Update()

        data = vtk_to_numpy(
            self.reader.GetOutput().GetPointData().GetScalars()
        ).reshape(self.reader.GetOutput().GetDimensions(), order='F')

        if convert_to_density:
            m, b = get_aim_density_equation(self.reader.GetProcessingLog())
            data = m * data + b

        position = list(self.reader.GetPosition())

        return data, position

    def _get_processing_log(self, filename):

        self.reader.SetFileName(filename)
        self.reader.Update()

        return self.reader.GetProcessingLog()

    def _get_spacing_and_origin(self, filename):

        self.reader.SetFileName(filename)
        self.reader.Update()

        return self.reader.GetOutput().GetSpacing(), self.reader.GetOutput().GetOrigin()