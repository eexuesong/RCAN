# RCAN

This repository is an updated and detailed version of repository [3D-RCAN](https://github.com/AiviaCommunity/3D-RCAN).

## System Requirements

- Windows 10. Linux and Mac OS should be able to run the code but the code has been only tested on Windows 10 & 11 so far.
- Python 3.7+
- NVIDIA GPU
- CUDA 10.0 and cuDNN 7.6.5

Tested Environment:

- Windows 11
- Python 3.10
- NVIDIA RTX A5000 24 GB
- CUDA 11.2 and cuDNN 8.1.0

## Environment Configuration:
1. Install [Anaconda](https://www.anaconda.com/download) and [Pycharm](https://www.jetbrains.com/pycharm/download/#section=windows).

2. Create a conda environment.
    - In Pycharm, create a new project named e.g. "RCAN3D" using [Conda environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html). A new environment with same name will also be created.
    - Or in [Anaconda Prompt](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html), create a new environment named "RCAN3D" by:
        ```posh
        conda create --name RCAN3D python=3.10
        ```
        To see a list of all your environments, type:
        ```posh
        conda info --envs
        ```
        or:
        ```posh
        conda env list
        ```
        Then in Pycharm, create a new project named e.g. "RCAN3D" using Conda environment.
        Select the location of the Conda environment we just created as:
        ```posh
        C:\Users\username\AppData\Local\anaconda3\envs\RCAN3D
        ```

3. In Anaconda Prompt, activate the new environment:
    ```posh
    conda activate RCAN3D
    ```

4. You should see (RCAN3D) in the command line.

## Package Installation:
5. GPU setup
    
    Note: Important, or you will run DL on CPU instead. Without a GPU, training and prediction will be much slower (~30-60 times, even when using a computer with 40 CPU cores):
    
    - First install NVIDIA GPU driver if you have not.

    - For GPU support, it is very important to install the specific versions of CUDA and cuDNN that are compatible with the respective version of TensorFlow.
        
        Install the CUDA (11.2), cuDNN (8.1.0) ([Question about version selection?](https://www.tensorflow.org/install/source_windows#gpu:)):
        ```posh
        conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
        ```

6. Assuming your project is located in: D:\Code\Python Code\RCAN3D, the "requirements.txt" must be placed in the same folder. In Anaconda Promp, run:
    ```posh
    D:
    cd D:\Code\Python Code\RCAN3D
    pip install -r requirements.txt
    ```

7. In Anaconda Promp, verify the GPU setup:
    ```posh
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```
    
## Training:
Training a 3D-RCAN model requires a [`config.json`](config.json) file to specify parameters and data locations.

To initiate training, use the command:
    ```posh
    python train.py -c config.json -o /path/to/training/output/dir
    ```
    
Training data can be specified in the config.json file by either:

Providing paths to folders containing raw and ground truth images (training_data_dir).
Listing specific pairs of raw and ground truth image files (training_image_pairs).
Numerous other options can be configured in the JSON file, including validation data, model architecture parameters (like num_residual_blocks, num_residual_groups), data augmentation settings, learning rate, loss functions, and metrics. The defaults for num_residual_blocks (3) and num_residual_groups (5) are set to balance performance and hardware constraints, aiming for optimal accuracy on standard GPUs (16-24GB VRAM) without causing memory overflow.

The user must specify the training data location in the input config JSON file to load the training images. We provide two ways to do so:

### (Option 1) Load images from a folder using `training_data_dir`

```javascript
"training_data_dir": {"raw":"/path/to/training/Raw/",
                      "gt":"/path/to/training/GT/"}
```

If use option 1, please make sure that Raw and GT directories contain the same number of TIFF files. TIFF files in raw and GT directories are sorted in alphabetical order by name when matching the raw/GT pairs. The file names of each raw/GT pair are output in the terminal window when loading data. Please check the output to make sure raw and GT are correctly matched.

### (Option 2) Load specific raw/grountruth image pairs using `training_image_pairs`

```javascript
"training_image_pairs": [
      {"raw": "/path/to/training/Raw/image1.tif",
        "gt": "/path/to/training/GT/image1.tif"},
      {"raw": "/path/to/training/Raw/image2.tif",
        "gt": "/path/to/training/GT/image2.tif"}
]
```

If use option 2, training data is an array of raw and GT image pairs.

Note that you can also use `training_data_dir` and `training_image_pairs` at the same time.

```javascript
"training_data_dir": {"raw":"/path/to/training/Raw/",
                      "gt":"/path/to/training/GT/"},
"training_image_pairs": [
     {"raw": "/path/to/additional/Raw/image1.tif",
      "gt": "/path/to/additional/GT/image1.tif"},
     {"raw": "/path/to/additional/Raw/image2.tif",
      "gt": "/path/to/additional/GT/image2.tif"}
]
```

### More options

Following optional variables can be also set in the JSON file (if not set, default values will be used):

- `validation_data_dir`
  
  - Paths to raw and groud truth data directories for validation
    
  - Default: None
  
    ```javascript
    "validation_data_dir": {"raw":"/path/to/validation/Raw/",
                            "gt":"/path/to/validation/GT/"}
    ```
  
- `validation_image_pairs` (array of image pairs)
  
  - Validation data on which to evaluate the loss and metrics at the end of each epoch

  - Default: None
  
    ```javascript
    "validation_image_pairs": [
         {"raw": "/path/to/additional/Raw_validation/image1.tif",
          "gt": "/path/to/additional/GT_validation/image1.tif"},
         {"raw": "/path/to/additional/Raw_validation/image2.tif",
          "gt": "/path/to/additional/GT_validation/image2.tif"}
        ]
    ```

## Notes:
 (1) Do the following before initializing TensorFlow to limit TensorFlow to first GPU:
 
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

 (2) You can find out which version of TensorFlow is installed via:

    pip show tensorflow

