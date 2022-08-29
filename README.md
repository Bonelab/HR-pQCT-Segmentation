# HR-pQCT-Segmentation

## Environment set-up

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
conda create -n blptl -c numerics88 -c conda-forge pytorch torchvision h5py pytorch-lightning torchmetrics scikit-learn pandas scipy matplotlib jupyterlab n88tools vtk simpleitk scikit-image
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
