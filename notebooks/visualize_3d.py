from __future__ import annotations

from visualization_utils import ImageType, generate_aim_spec, align_aims, read_image_to_numpy

from bonelab.util.aim_calibration_header import get_aim_density_equation
from bonelab.util.vtk_util import vtkImageData_to_numpy
from vtkbone import vtkboneAIMReader
from enum import Enum
import numpy as np
import pyvista as pv
import os

# constants
CORT_THRESH = 450
TRAB_THRESH = 320


def read_data_to_pyvista(reader: vtkboneAIMReader, image_name: str, aim_spec: Dict[Dict[str]]) -> pv.UniformGrid:
    keys = []
    data = []
    for key, aim in aim_spec.items():
        keys.append(key)
        filename = os.path.join(aim["dir"], f"{image_name}{aim['suffix']}")
        data.append(read_image_to_numpy(reader, filename, aim["type"]))
    data = align_aims(data)
    grid = pv.UniformGrid(dims=data[0].shape, spacing=reader.GetOutput().GetSpacing())
    for key, image in zip(keys, data):
        grid[key] = image.ravel(order="F")
    return grid


def render_cortical_masks():
    image_dir = "/Users/nathanneeteson/Documents/Data/Images/UNet"
    preds_dir = "/Users/nathanneeteson/Documents/Data/Images/UNet/predictions"
    image_names = ["NORMXTII_0051_TL", "NORMXTII_0057_TL"]
    aim_spec = generate_aim_spec(image_dir, preds_dir)
    # read in data
    reader = vtkboneAIMReader()
    reader.DataOnCellsOff()
    grids = {}
    for image_name in image_names:
        print(f"Loading data for {image_name}")
        grids[image_name] = read_data_to_pyvista(reader, image_name, aim_spec)
        grids[image_name]["cort_mask_disagreement"] = (
            np.abs(grids[image_name]["cort_mask_reference"] - grids[image_name]["cort_mask_predicted"])
        )
    n_rows = 2
    n_cols = 2
    smooth_n_iter = 100
    plotter = pv.Plotter(
        window_size=(600, 600),
        shape=(n_rows, n_cols),
        lighting='three lights'
    )
    plotter.set_background("white")
    mask_name = ["cort_mask_reference", "cort_mask_predicted"]
    color = "silver"
    for row in range(n_rows):
        for col in range(n_cols):
            print(f"Plotting {image_names[col]}, {mask_name[row]}")
            plotter.subplot(row, col)
            plotter.add_mesh(
                grids[image_names[col]]
                .contour([0.5], scalars=mask_name[row], progress_bar=True)
                .smooth(n_iter=smooth_n_iter, progress_bar=True),
                color=color
            )
    plotter.link_views()
    plotter.show()


def render_bone():
    image_dir = "/Users/nathanneeteson/Documents/Data/Images/UNet"
    preds_dir = "/Users/nathanneeteson/Documents/Data/Images/UNet/predictions"
    image_names = ["NORMXTII_0051_TL", "NORMXTII_0057_TL"]
    aim_spec = generate_aim_spec(image_dir, preds_dir)
    # read in data
    reader = vtkboneAIMReader()
    reader.DataOnCellsOff()
    grids = {}
    for image_name in image_names:
        print(f"Loading data for {image_name}")
        grids[image_name] = read_data_to_pyvista(reader, image_name, aim_spec)
        grids[image_name]["cort_mask_disagreement"] = (
            np.abs(grids[image_name]["cort_mask_reference"] - grids[image_name]["cort_mask_predicted"])
        )
    n_rows = 2
    n_cols = 2
    smooth_n_iter = 100
    plotter = pv.Plotter(
        window_size=(600, 600),
        shape=(n_rows, n_cols),
        lighting='three lights'
    )
    plotter.set_background("white")
    cort_mask_name = ["cort_mask_reference", "cort_mask_predicted"]
    trab_mask_name = ["trab_mask_reference", "trab_mask_predicted"]
    cort_color = "silver"
    trab_color = "slategrey"
    for row in range(n_rows):
        for col in range(n_cols):
            print(f"Plotting {image_names[col]}, {cort_mask_name[row]}")
            plotter.subplot(row, col)
            plotter.add_mesh(
                grids[image_names[col]]
                .threshold(0.5, scalars=cort_mask_name[row], progress_bar=True)
                .contour([CORT_THRESH], scalars="image", progress_bar=True)
                .smooth(n_iter=smooth_n_iter, progress_bar=True),
                color=cort_color
            )
            print(f"Plotting {image_names[col]}, {trab_mask_name[row]}")
            plotter.add_mesh(
                grids[image_names[col]]
                .threshold(0.5, scalars=trab_mask_name[row], progress_bar=True)
                .contour([TRAB_THRESH], scalars="image", progress_bar=True)
                .smooth(n_iter=smooth_n_iter, progress_bar=True),
                color=trab_color
            )
    plotter.link_views()
    plotter.show()


def main():
    #render_cortical_masks()
    render_bone()


if __name__ == "__main__":
    main()
