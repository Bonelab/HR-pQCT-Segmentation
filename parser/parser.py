import argparse

def check_odd_integer(value):
    ivalue = int(value)
    if not(ivalue%2):
        raise argparse.ArgumentTypeError(f"{ivalue} is not a valid odd integer")
    return ivalue

# PARSER CREATION FUNCTION
def create_parser():

    parser = argparse.ArgumentParser(
        description='HRpQCT Segmentation 2D UNet Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--label', type=str, default='U-Net-2D', metavar='STR',
        help='base label for output files'
    )
    parser.add_argument(
        '--log-dir', type=str, default='./logs/', metavar='STR',
        help='path of directory to save log to'
    )
    parser.add_argument(
        '--training-data-dir', type=str, default='./data/training/', metavar='STR',
        help='path of directory containing training data '
    )
    parser.add_argument(
        '--validation-data-dir', type=str, default='./data/validation/', metavar='STR',
        help='path of directory containing validation data'
    )
    parser.add_argument(
        '--image-loader-workers', type=int, default=0, metavar='N',
        help='number of cpu workers loading images from file'
    )
    parser.add_argument(
        '--slice-loader-workers', type=int, default=0, metavar='N',
        help='number of cpu workers getting slices from images'
    )
    parser.add_argument(
        '--min-density', type=float, default=-400, metavar='D',
        help='minimum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        '--max-density', type=float, default=1400, metavar='D',
        help='maximum physiologically relevant density in the image [mg HA/ccm]'
    )
    parser.add_argument(
        '--trained-model-dir', type=str, default='./trained_models/', metavar='STR',
        help='path of directory to save trained model to'
    )
    parser.add_argument(
        '--prev-trained-model', type=str, default=None, metavar='STR',
        help='path to previously trained model to start model parameters at'
    )
    parser.add_argument(
        '--optimizer-dir', type=str, default='./optimizer_state_dicts/', metavar='STR',
        help='path of directory to save optimizer state dicts to'
    )
    parser.add_argument(
        '--prev-optimizer', type=str, default=None, metavar='STR',
        help='path to previously used optimizer to maintain mom/RMS'
    )
    parser.add_argument(
        '--model-filters', type=int, nargs='+', default=[32, 64, 128, 256], metavar='N',
        help='sequence of filters in U-Net layers'
    )
    parser.add_argument(
        '--channels-per-group', type=int, default=16, metavar='N',
        help='channels per group in GroupNorm'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, metavar='D',
        help='dropout probability'
    )
    parser.add_argument(
        '--lambda-curvature', type=float, default=1e-5, metavar='D',
        help='curvature regularization coefficient'
    )
    parser.add_argument(
        '--lambda-maggrad', type=float, default=1e-5, metavar='D',
        help='magnitude gradient curvature coefficient'
    )
    parser.add_argument(
        '--curvature-threshold', type=float, default=0.005, metavar='D',
        help='maximum curvature above which excess curvature will be penalized, units: 1/um'
    )
    parser.add_argument(
        '--voxel-width', type=float, default=61, metavar='D',
        help='isotropic voxel width, units: um'
    )
    parser.add_argument(
        '--heaviside-epsilon', type=float, default=0.1, metavar='D',
        help='scaling parameter for the approximate heaviside function'
    )
    parser.add_argument(
        '--num-epochs-half-cycle', type=int, default=10, metavar='N',
        help='number of epochs in half of the main cycle'
    )
    parser.add_argument(
        '--num-epochs-convergence', type=int, default=5, metavar='N',
        help='number of epochs in the convergence phase'
    )
    parser.add_argument(
        '--starting-epoch', type=int, default=1, metavar='N',
        help='if resuming previous training, epoch to start at'
    )
    parser.add_argument(
        '--stopping-epoch', type=int, default=None, metavar='N',
        help='epoch to stop at, if stopping early'
    )
    parser.add_argument(
        '--training-batch-size', type=int, default=3, metavar='N',
        help='training batch size'
    )
    parser.add_argument(
        '--validation-batch-size', type=int, default=30, metavar='N',
        help=' validation batch size'
    )
    parser.add_argument(
        '--opt-min-lr', type=float, default=1e-4, metavar='LR',
        help='minimum learning rate, as determined by range plot analysis'
    )
    parser.add_argument(
        '--opt-max-lr', type=float, default=1e-3, metavar='LR',
        help='maximum learning rate, as determined by range plot analysis'
    )
    parser.add_argument(
        '--opt-min-momentum', type=float, default=0.85, metavar='M',
        help='minimum momentum coefficient for AdamW'
    )
    parser.add_argument(
        '--opt-max-momentum', type=float, default=0.95, metavar='M',
        help='maximum momentum coefficient for AdamW'
    )
    parser.add_argument(
        '--opt-rms', type=float, default=0.999, metavar='RMS',
        help='rms coefficient for AdamW'
    )
    parser.add_argument(
        '--opt-weight_decay', type=float, default=0, metavar='WD',
        help='weight decay regularization coefficient for AdamW'
    )
    parser.add_argument(
        '--opt-eps', type=float, default=1e-8, metavar='EPS',
        help='epsilon for AdamW'
    )
    parser.add_argument(
        '--input-channels', type=check_odd_integer, default=5, metavar='N',
        help='number slices to use as input for each slice prediction, must be odd'
    )
    parser.add_argument(
        '--output-channels', type=int, default=2, metavar='N',
        help='number of channels in the output data'
    )
    parser.add_argument(
        '--no-clamp-gradients', action='store_true', default=False,
        help='disable gradient clamping during training'
    )
    parser.add_argument(
        '--cuda', action='store_true', default=False,
        help='enable cuda processing'
    )
    parser.add_argument(
        '--device-ids', type=int, nargs='+', default=[0], metavar='N',
        help='device ids for devices to use in training, CPU/GPU. If more than 1 given, DataParallel used'
    )
    parser.add_argument(
        '--dry-run', action='store_true', default=False,
        help='quick single pass through'
    )

    return parser
