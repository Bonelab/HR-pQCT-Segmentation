import os
import torch
import yaml
import vtk


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from dataset.SamplePadder import SamplePadder
from dataset.SampleStandardizer import SampleStandardizer
from dataset.SampleToTensors import SampleToTensors
from dataset.NiftiDataset import NiftiDataset

from utils.postprocessing import postprocess_masks_iterative
from utils.image_export import save_masks_as_niftis

from models.UNet import UNet

from dataset.HRpQCTAIMDataset import HRpQCTAIMDataset

from traintest.infer import infer

VERSION = "1.0"


def create_parser():
    parser = ArgumentParser(
        description='HRpQCT Segmentation 2D UNet Segmenting Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "image_directory", type=str, metavar="DIR",
        help="directory containing AIM images to segment"
    )
    parser.add_argument(
        "model_label", type=str, metavar="LABEL",
        help=(
            "the label of the trained model to use for segmenting - corresponding *.pth and *.yaml files "
            "must be in /trained_models subdirectory. "
            "create these files by training a new model on your own data or email skboyd@ucalgary.ca "
            "to request the published model for radius and tibia images"
        )
    )
    parser.add_argument(
        "--image-pattern", "-ip", type=str, default="*.nii", metavar="STR",
        help="`glob`-compatible pattern to match to find your images in the directory"
    )
    parser.add_argument(
        "--masks-subdirectory", "-ms", type=str, default="masks", metavar="STR",
        help="subdirectory, inside of `image-directory`, to save the masks to"
    )
    parser.add_argument(
        "--cuda", "-c", action="store_true",
        help="enable this flag to use CUDA / GPU"
    )

    return parser


def main():
    # parse the arguments
    args = create_parser().parse_args()

    # check what device to use
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    if args.cuda and not torch.cuda.is_available():
        print("WARNING: cuda flag was set, but cuda is not available! using cpu...")

    # create the masks' subdirectory if it doesn't already exist
    try:
        os.mkdir(os.path.join(args.image_directory, args.masks_subdirectory))
    except FileExistsError:
        pass  # if the directory already exists, that's fine

    # load the hyperparameters from the training run
    hyperparameters_file = os.path.join(".", "trained_models", f"{args.model_label}.yaml")
    try:
        with open(hyperparameters_file, 'r') as f:
            try:
                hparams = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The trained model hyperparameter yaml file was not found: {e}")

    # create the UNet
    model = UNet(hparams["input_channels"], hparams["output_channels"],
                 hparams["model_filters"], hparams["channels_per_group"], hparams["dropout"])
    model.float()
    model.to(device)

    # load the trained model parameters
    model_parameters_file = os.path.join(".", "trained_models", f"{args.model_label}.pth")
    try:
        model.load_state_dict(torch.load(model_parameters_file, map_location=device))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The trained model parameters pth file was not found: {e}")

    # put the model in evaluation mode
    model.eval()

    # create transforms consistent with how images were transformed during training
    # create dataset transforms
    transforms = Compose([
        SamplePadder(2 ** (len(hparams["model_filters"]) - 1)),
        SampleStandardizer(hparams["min_density"], hparams["max_density"]),
        SampleToTensors(ohe=False)
    ])

    # create dataset
    dataset = NiftiDataset(args.image_directory, args.image_pattern, transform=transforms, load_masks=False)
    print(f"# images to be segmented: {len(dataset)}")

    # create kwargs for dataloader
    dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False
    }

    # create kwargs for dataset and dataloader for single images
    image_dataset_kwargs = {
        'num_adjacent_slices': (hparams["input_channels"] - 1) // 2
    }
    image_dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False
    }

    # create the overall dataloader
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    for image in dataloader:
        # extract the name of the image
        image_name = os.path.splitext(image['name'][0])[0]
        print(f"Segmenting {image_name}...")
        # and extract the image data for use in post-processing
        image_data = image['image'][0, 0, :, :, :].cpu().detach().numpy()
        # get the embeddings from the model
        print("- predicting embeddings... ", end="")
        start_time = timer()
        phi_peri, phi_endo = infer(
            None, model, device, image,
            image_dataset_kwargs, image_dataloader_kwargs
        )
        print(f"done! ({timer()-start_time:0.3f} s)")
        # convert embeddings to masks
        print("- converting embeddings to masks... ", end="")
        start_time = timer()
        cort_mask = (phi_peri < 0) * (phi_endo > 0)
        trab_mask = phi_endo < 0
        print(f"done! ({timer()-start_time:0.3f} s)")
        # post-process masks
        print("- post-processing masks... ", end="")
        start_time = timer()
        cort_mask, trab_mask = postprocess_masks_iterative(
            image_data, cort_mask, trab_mask, visualize=False
        )
        print(f"done! ({timer()-start_time:0.3f} s)")
        # save the masks
        print("- writing masks to file... ", end="")
        start_time = timer()
        save_masks_as_niftis(
            os.path.join(
                args.image_directory,
                args.masks_subdirectory,
                f"{image_name}_mask.nii.gz"
            ),
            cort_mask + 2 * trab_mask,
            image['image_position'],
            image['image_position_original'],
            image['image_shape_original'],
            dataset.get_vtk_image_data()
        )
        print(f"done! ({timer()-start_time:0.3f} s)")


if __name__ == "__main__":
    main()
