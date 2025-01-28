# HR-pQCT-Segmentation

## Citing

Neeteson, N.J., Besler, B.A., Whittier, D.E. et al. Automatic segmentation of trabecular and cortical compartments in HR-pQCT images using an embedding-predicting U-Net and morphological post-processing. Sci Rep 13, 252 (2023). https://doi.org/10.1038/s41598-022-27350-0

## Model weights

Zenodo: https://zenodo.org/records/14755838

## 0.0 Environment set-up

1. Go to Projects directory (or wherever you store git repos).
```
cd ~/Projects
```
2. Clone the bonelab repo.
```
git clone https://github.com/Bonelab/Bonelab.git
```
3. Create the bonelab + pytorch environment.

With GPU:
```
conda create -n bl_torch -c numerics88 -c simpleitk -c conda-forge -c pytorch python=3.7 n88tools pbr nose six simpleitk pydicom gdcm pytorch-gpu torchvision scikit-image
```

Without GPU:
```
conda create -n bl_torch -c numerics88 -c simpleitk -c conda-forge -c pytorch python=3.7 n88tools pbr nose six simpleitk pydicom gdcm pytorch torchvision scikit-image
```

With lightning and jupyterlab (sub pytorch-gpu for pytorch if you want):
```
conda create -n bl_torch -c numerics88 -c conda-forge pytorch torchvision h5py pytorch-lightning torchmetrics scikit-learn pandas scipy matplotlib jupyterlab n88tools vtk simpleitk scikit-image
```

4. Activate the environment.
```
conda activate bl_torch
```
5. Go into the bonelab repo.
```
cd Bonelab
```
6. Install the additional modules and cli entry points.
```
pip install -e .
```
7. Clone this repo.
```
cd ~/Projects
git clone git@github.com:Bonelab/HR-pQCT-Segmentation.git
```

Now you can run scripts from this repo from the cloned repo directory.

## 1.0 Segmenting AIMs

The script `segment.py` can be used to segment a directory of AIMs using a trained model. 

To get more usage instructions, type: `python segment.py -h`:

```
(blptl) HR-pQCT-Segmentation % python segment.py -h
usage: segment.py [-h] [--image-pattern STR] [--masks-subdirectory STR]
                  [--cuda]
                  DIR LABEL

HRpQCT Segmentation 2D UNet Segmenting Script

positional arguments:
  DIR                   directory containing AIM images to segment
  LABEL                 the label of the trained model to use for segmenting -
                        corresponding *.pth and *.yaml files must be in
                        /trained_models subdirectory. create these files by
                        training a new model on your own data or email
                        skboyd@ucalgary.ca to request the published model for
                        radius and tibia images

optional arguments:
  -h, --help            show this help message and exit
  --image-pattern STR, -ip STR
                        `glob`-compatible pattern to match to find your images
                        in the directory (default: *_*_??.AIM)
  --masks-subdirectory STR, -ms STR
                        subdirectory, inside of `image-directory`, to save the
                        masks to (default: masks)
  --cuda, -c            enable this flag to use CUDA / GPU (default: False)

```

The trained model parameters files used to produce the segmentations in the publication are not provided in this repository. If you want to use this segmentation program, please email skboyd@ucalgary.ca to request that the files be transferred to you.

A slurm file (`slurm/segment.slurm`) and shell script (`segment_arc.sh`) have also been provided to make it easier to use ARC to segment images. The shell script runs a slurm job using the slurm file and requires that the name of the `conda` environment, image directory, and trained model label are passed to it as command line arguments:
```
./segment_arc.sh
--------------------
Required arguments:
--------------------
Argument 1: name of the conda environment to use, e.g. `blptl`
Argument 2: path to the directory that contains the images to segment, e.g. `./data/images`
Argument 3: the label of the trained model to use to segment, e.g. `radius_and_tibia_final`
--------------------
Example usage: ./segment_arc.sh blptl ./data/images radius_and_tibia_final
```

Steps:
1. Transfer yopur images to ARC (e.g. using `scp` or FileZilla)
2. Set up your environment on ARC, following the ARC instructions for setting up `miniconda` and then following the directions in 0.0 for setting up the `conda` environment you need to use to run the scripts in this repository.
3. Clone this repository to somewhere in your ARC home directory (e.g. `$HOME/Repositories/HR-pQCT-Segmentation`)
4. Obtain the `*.pth` and `*.yaml` files for a trained model and place them in a sub-directory called `trained_models` (e.g. `$HOME/Repositories/HR-pQCT-Segmentation/trained_models/`)
5. Enable execution permission for `segment_arc.sh`: `chmod +x segment_arc.sh`
6. Run `segment_arc.sh` with the appropriate command line arguments to point it at the appropriate `conda` env, your images, and the trained model.
7. The segmentations will be placed in a `mask` sub-directory of the image directory.

## 2.0 Training a Model

The script `train.py` can be used to train a new model with your own data. 

To get more usage instructions, type: `python train.py -h`:

```
(blptl) HR-pQCT-Segmentation % python train.py -h
usage: train.py [-h] [--label STR] [--log-dir STR] [--training-data-dir STR]
                [--validation-data-dir STR] [--image-loader-workers N]
                [--slice-loader-workers N] [--min-density D] [--max-density D]
                [--trained-model-dir STR] [--prev-trained-model STR]
                [--optimizer-dir STR] [--prev-optimizer STR]
                [--model-filters N [N ...]] [--channels-per-group N]
                [--dropout D] [--lambda-curvature D] [--lambda-maggrad D]
                [--curvature-threshold D] [--voxel-width D]
                [--heaviside-epsilon D] [--num-epochs-half-cycle N]
                [--num-epochs-convergence N] [--starting-epoch N]
                [--stopping-epoch N] [--training-batch-size N]
                [--validation-batch-size N] [--opt-min-lr LR]
                [--opt-max-lr LR] [--opt-min-momentum M]
                [--opt-max-momentum M] [--opt-rms RMS] [--opt-weight_decay WD]
                [--opt-eps EPS] [--input-channels N] [--output-channels N]
                [--no-clamp-gradients] [--cuda] [--device-ids N [N ...]]
                [--dry-run]

HRpQCT Segmentation 2D UNet Training Script

optional arguments:
  -h, --help            show this help message and exit
  --label STR           base label for output files (default: U-Net-2D)
  --log-dir STR         path of directory to save log to (default: ./logs/)
  --training-data-dir STR
                        path of directory containing training data (default:
                        ./data/training/)
  --validation-data-dir STR
                        path of directory containing validation data (default:
                        ./data/validation/)
  --image-loader-workers N
                        number of cpu workers loading images from file
                        (default: 0)
  --slice-loader-workers N
                        number of cpu workers getting slices from images
                        (default: 0)
  --min-density D       minimum physiologically relevant density in the image
                        [mg HA/ccm] (default: -400)
  --max-density D       maximum physiologically relevant density in the image
                        [mg HA/ccm] (default: 1400)
  --trained-model-dir STR
                        path of directory to save trained model to (default:
                        ./trained_models/)
  --prev-trained-model STR
                        path to previously trained model to start model
                        parameters at (default: None)
  --optimizer-dir STR   path of directory to save optimizer state dicts to
                        (default: ./optimizer_state_dicts/)
  --prev-optimizer STR  path to previously used optimizer to maintain mom/RMS
                        (default: None)
  --model-filters N [N ...]
                        sequence of filters in U-Net layers (default: [32, 64,
                        128, 256])
  --channels-per-group N
                        channels per group in GroupNorm (default: 16)
  --dropout D           dropout probability (default: 0.1)
  --lambda-curvature D  curvature regularization coefficient (default: 1e-05)
  --lambda-maggrad D    magnitude gradient curvature coefficient (default:
                        1e-05)
  --curvature-threshold D
                        maximum curvature above which excess curvature will be
                        penalized, units: 1/um (default: 0.005)
  --voxel-width D       isotropic voxel width, units: um (default: 61)
  --heaviside-epsilon D
                        scaling parameter for the approximate heaviside
                        function (default: 0.1)
  --num-epochs-half-cycle N
                        number of epochs in half of the main cycle (default:
                        10)
  --num-epochs-convergence N
                        number of epochs in the convergence phase (default: 5)
  --starting-epoch N    if resuming previous training, epoch to start at
                        (default: 1)
  --stopping-epoch N    epoch to stop at, if stopping early (default: None)
  --training-batch-size N
                        training batch size (default: 3)
  --validation-batch-size N
                        validation batch size (default: 30)
  --opt-min-lr LR       minimum learning rate, as determined by range plot
                        analysis (default: 0.0001)
  --opt-max-lr LR       maximum learning rate, as determined by range plot
                        analysis (default: 0.001)
  --opt-min-momentum M  minimum momentum coefficient for AdamW (default: 0.85)
  --opt-max-momentum M  maximum momentum coefficient for AdamW (default: 0.95)
  --opt-rms RMS         rms coefficient for AdamW (default: 0.999)
  --opt-weight_decay WD
                        weight decay regularization coefficient for AdamW
                        (default: 0)
  --opt-eps EPS         epsilon for AdamW (default: 1e-08)
  --input-channels N    number slices to use as input for each slice
                        prediction, must be odd (default: 5)
  --output-channels N   number of channels in the output data (default: 2)
  --no-clamp-gradients  disable gradient clamping during training (default:
                        False)
  --cuda                enable cuda processing (default: False)
  --device-ids N [N ...]
                        device ids for devices to use in training, CPU/GPU. If
                        more than 1 given, DataParallel used (default: [0])
  --dry-run             quick single pass through (default: False)

```

Most hyperparameter defaults have been set at the values used in the publication. This script trains a single model and logs performance metrics in a `logs` sub-directory and stores the hyperparameters and model parameters in a `trained_models` sub-directory. The model is checkpointed at the end of each epoch. Hyper-parameter sweep / optimization scripts are not provided - if you want to do these kinds of experiments you'll have to write your own additional scripts.
