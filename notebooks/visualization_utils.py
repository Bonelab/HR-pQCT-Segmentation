from __future__ import annotations

from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.util.vtk_util import vtkImageData_to_numpy
import numpy as np
from enum import Enum

# create an ImageType Enum with two options: density image or mask
ImageType = Enum("ImageType", "DENSITY MASK")


def generate_aim_spec(image_dir: str, preds_dir: str) -> Dict[Dict[str]]:
    return {
        "image": {
            "dir": image_dir,
            "suffix": ".AIM",
            "type": ImageType.DENSITY
        },
        "cort_mask_reference": {
            "dir": image_dir,
            "suffix": "_CORT_MASK.AIM",
            "type": ImageType.MASK
        },
        "trab_mask_reference": {
            "dir": image_dir,
            "suffix": "_TRAB_MASK.AIM",
            "type": ImageType.MASK
        },
        "cort_mask_predicted": {
            "dir": preds_dir,
            "suffix": "_CORT_MASK.AIM",
            "type": ImageType.MASK
        },
        "trab_mask_predicted": {
            "dir": preds_dir,
            "suffix": "_TRAB_MASK.AIM",
            "type": ImageType.MASK
        }
    }


def align_aims(data: List[Tuple[np.ndarray, List[int]]]) -> List[np.ndarray]:
    min_position = np.asarray([p for _, p in data]).min(axis=0)
    pad_lower = [p - min_position for _, p in data]
    max_shape = np.asarray([(aim.shape + pl) for (aim, _), pl in zip(data, pad_lower)]).max(axis=0)
    pad_upper = [(max_shape - (aim.shape + pl)) for (aim, _), pl in zip(data, pad_lower)]
    return [
        np.pad(aim, tuple([(l, u) for l, u in zip(pl, pu)]), "constant")
        for (aim, _), pl, pu in zip(data, pad_lower, pad_upper)
    ]


def read_image_to_numpy(reader: vtkboneAIMReader, filename: str, image_type: ImageType) -> Tuple[np.ndarray, list]:
    reader.SetFileName(filename)
    reader.Update()
    data = vtkImageData_to_numpy(reader.GetOutput())
    if image_type is ImageType.DENSITY:
        # convert data to densities
        m, b = get_aim_density_equation(reader.GetProcessingLog())
        data = m * data + b
    elif image_type is ImageType.MASK:
        # convert data to a binary mask
        data = (data > 0).astype(int)
        # pad the axial ends so the surfaces close
        data = np.pad(data, ((0, 0), (0, 0), (1, 1)), "constant")
    return data, reader.GetPosition()


def single_axis_image_and_masks(ax, image, masks, colors, title=None):
    # ax: the plot axis to plot things on
    # image: a 2D gray-scale image as an array
    # masks: a list of binary masks as 2D arrays
    # colors: a list of colormap colors corresponding to masks
    # title: a string, to put over the plot

    ax.imshow(
        image,
        cmap='gray'
    )

    for mask, color in zip(masks,colors):

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
