"""
Written by Nathan Neeteson.
Training a U-Net on dataset of segmented images.
"""

# IMPORTS
import os
import sys
from datetime import datetime

import torch
from torch.nn import DataParallel
from torch.optim import AdamW
from torchvision.transforms import Compose

from models.unet import UNet
from utils.error_metrics import (
    HRpQCTEmbeddingNLLLoss, CurvatureLoss, MagnitudeGradientSDTLoss,
    HRpQCTEmbeddingCombinedRegularizationLoss,
    create_calculate_embedding_dice_coefficient
)
from utils.bone_contouring_dataset import (
    SamplePadder, SampleToTensors, SampleStandardizer,
    HRpQCT_AIM_Dataset, SingleImageDataset
)
from utils.optimizer_scheduling import OptimizerSchedulerLinear
from utils.logging import Logger

from parser.parser import create_parser
from traintest.traintest import traintest


# MAIN FUNCTION
def main():
    # Settings
    parser = create_parser()
    args = parser.parse_args()

    # create trained model and logs directories if necessary
    for d in [args.log_dir, args.trained_model_dir]:
        if not (os.path.isdir(d)):
            os.mkdir(d)

    # assemble filenames for the trained model and the log file
    model_filename = f'{args.trained_model_dir}{args.label}.pth'
    log_filename = f'{args.log_dir}{args.label}.csv'
    optimizer_filename = f'{args.optimizer_dir}{args.label}.pth'

    # check what device to use
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    # create the model
    model = UNet(args.input_channels, args.output_channels,
                 args.model_filters, args.channels_per_group, args.dropout)
    model.float()
    model.to(device)

    # load the previous model parameters, if given
    if args.prev_trained_model:
        model.load_state_dict(torch.load(args.prev_trained_model))

    # wrap the model in a parallel processing module if using multiple devices
    if len(args.device_ids) > 1:
        print('Using Data Parallel')
        model = DataParallel(model, device_ids=args.device_ids)

    # create dataset transforms
    data_transforms = Compose([
        SamplePadder(2 ** (len(args.model_filters) - 1)),
        SampleStandardizer(args.min_density, args.max_density),
        SampleToTensors(ohe=False)
    ])

    # create datasets
    training_dataset = HRpQCT_AIM_Dataset(args.training_data_dir, transform=data_transforms)
    validation_dataset = HRpQCT_AIM_Dataset(args.validation_data_dir, transform=data_transforms)

    # construct dictionary of testing functions
    testing_functions = {
        'dice': create_calculate_embedding_dice_coefficient(args.heaviside_epsilon)
    }

    # create optimizer kwards dict
    optimizer_kwargs = {
        'lr': args.opt_min_lr,
        'betas': (args.opt_max_momentum, args.opt_rms),
        'eps': args.opt_eps,
        'weight_decay': args.opt_weight_decay
    }

    # create optimizer
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)

    # load the optimizer state dict, if given
    if args.prev_optimizer:
        optimizer.load_state_dict(torch.load(args.prev_optimizer))

    # create the loss dictionary
    losses = {}

    losses['NLL'] = {}
    losses['NLL']['function'] = HRpQCTEmbeddingNLLLoss(args.heaviside_epsilon)
    losses['NLL']['coefficient'] = 1

    losses['Curvature'] = {}
    losses['Curvature']['function'] = \
        HRpQCTEmbeddingCombinedRegularizationLoss(
            CurvatureLoss(args.voxel_width, args.curvature_threshold, device)
        )
    losses['Curvature']['coefficient'] = args.lambda_curvature

    losses['MagGrad'] = {}
    losses['MagGrad']['function'] = \
        HRpQCTEmbeddingCombinedRegularizationLoss(
            MagnitudeGradientSDTLoss(args.voxel_width, device)
        )
    losses['MagGrad']['coefficient'] = args.lambda_maggrad

    # create training dataset and dataloader kwargs dicts
    training_dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': args.image_loader_workers
    }

    training_image_dataset_kwargs = {
        'num_adjacent_slices': (args.input_channels - 1) // 2
    }

    training_image_dataloader_kwargs = {
        'batch_size': args.training_batch_size,
        'shuffle': True,
        'num_workers': args.slice_loader_workers
    }

    # create validation dataset and dataloader kwargs dicts
    validation_dataloader_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': args.image_loader_workers
    }

    validation_image_dataset_kwargs = {
        'num_adjacent_slices': (args.input_channels - 1) // 2
    }

    validation_image_dataloader_kwargs = {
        'batch_size': args.validation_batch_size,
        'shuffle': False,
        'num_workers': args.slice_loader_workers
    }

    # create the optimizer scheduler
    optimizer_scheduler = OptimizerSchedulerLinear(
        [
            1,
            args.num_epochs_half_cycle,
            2 * args.num_epochs_half_cycle,
            2 * args.num_epochs_half_cycle + args.num_epochs_convergence
        ],
        [
            args.opt_min_lr,
            args.opt_max_lr,
            args.opt_min_lr,
            args.opt_min_lr / 10
        ],
        [
            args.opt_max_momentum,
            args.opt_min_momentum,
            args.opt_max_momentum,
            args.opt_max_momentum
        ]
    )

    # establish the list of fields to be logged
    log_fields = ['epoch', 'train/test', 'idx', 'name'] + list(testing_functions.keys())

    # add all of the losses
    for loss in losses.keys():
        log_fields.append(loss)

    # add all of the testing functions
    for testing_function in testing_functions.keys():
        log_fields.append(testing_function)

    # create the logger
    logger = Logger(log_filename, log_fields, args)

    num_epochs = args.stopping_epoch if args.stopping_epoch else 2 * args.num_epochs_half_cycle + args.num_epochs_convergence + 1

    # train and validate
    for epoch in range(args.starting_epoch, num_epochs):

        logger.set_field_value('epoch', epoch)

        # assign optimizer params according to schedule
        optimizer_scheduler.set_epoch(epoch)

        for g in optimizer.param_groups:
            g['lr'] = optimizer_scheduler.get_lr()
            g['momentum'] = optimizer_scheduler.get_mom()

        # train one epoch
        traintest(
            args, model, device,
            training_dataset, training_dataloader_kwargs,
            training_image_dataset_kwargs, training_image_dataloader_kwargs,
            optimizer, losses, testing_functions, logger, train=True
        )

        # validate one epoch
        traintest(
            args, model, device,
            validation_dataset, validation_dataloader_kwargs,
            validation_image_dataset_kwargs, validation_image_dataloader_kwargs,
            None, losses, testing_functions, logger, train=False
        )

        # checkpoint the model
        if len(args.device_ids) > 1:
            torch.save(model.module.state_dict(), model_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save(model.state_dict(), model_filename, _use_new_zipfile_serialization=False)

        # checkpoint the optimizer
        torch.save(optimizer.state_dict(), optimizer_filename, _use_new_zipfile_serialization=False)

        # stop iterating, if in a dry run test
        if args.dry_run:
            break


if __name__ == '__main__':
    main()
