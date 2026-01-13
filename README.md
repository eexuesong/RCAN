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

## Environment Configuration
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

## Package Installation
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
    
## Training
Training a 3D-RCAN model requires a [`config.json`](config.json) file to specify parameters and data locations.

To initiate training, use the command:
```posh
python train.py -c config.json -o /path/to/training/output/dir
```

Training data can be specified in the `config.json` file by either:
### (Option 1) Providing paths to folders containing raw/grountruth image pairs (`training_data_dir`).

```javascript
"training_data_dir": {"raw":"/path/to/training/Raw/",
                      "gt":"/path/to/training/GT/"}
```

If use option 1, please make sure that Raw and GT directories contain the same number of TIFF files. TIFF files in raw and GT directories are sorted in alphabetical order by name when matching the raw/GT pairs. The file names of each raw/GT pair are output in the terminal window when loading data. Please check the output to make sure raw and GT are correctly matched.

### (Option 2) Listing specific raw/grountruth image pairs (`training_image_pairs`).

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
Numerous other options can be configured in the JSON file (if not set, default values will be used), including validation data, model architecture parameters (like `num_residual_blocks`, `num_residual_groups`), data augmentation settings, learning rate, loss functions, and metrics. The defaults for `num_residual_blocks` (3) and `num_residual_groups` (5) are set to balance performance and hardware constraints, aiming for optimal accuracy on standard GPUs (16-24GB VRAM) without causing memory overflow.

- `validation_data_dir`: Paths to raw and groud truth data directories for validation (optional)
    
  - Default: None
  
    ```javascript
    "validation_data_dir": {"raw":"/path/to/validation/Raw/",
                            "gt":"/path/to/validation/GT/"}
    ```
  
- `validation_image_pairs`: Validation data on which to evaluate the loss and metrics at the end of each epoch (array of image pairs) (optional)

  - Default: None
  
    ```javascript
    "validation_image_pairs": [
         {"raw": "/path/to/additional/Raw_validation/image1.tif",
          "gt": "/path/to/additional/GT_validation/image1.tif"},
         {"raw": "/path/to/additional/Raw_validation/image2.tif",
          "gt": "/path/to/additional/GT_validation/image2.tif"}
        ]
    ```


- `epochs` (integer): Number of epochs to train the model
  - Default: 300,  Range: >=1  
    ```javascript
    "epochs": 100 ~ 300
    ```
  
- `steps_per_epoch` (integer): Number of steps to perform back-propagation on mini-batches in each epoch
  - Default: 256, Range: >=1
    ```javascript
    "steps_per_epoch": 100, 256 or 400
    ```
  
- `num_channels` (integer): Number of feature channels in RCAN
  - Default: 32, Range: >=1
    ```javascript
    "num_channels": 32
    ```
  
- `num_residual_blocks` (integer): Number of residual channel attention blocks in each residual group in RCAN
  - Default: 3, Range: >=1
    ```javascript
    "num_channels": 3, 4 or 5
    ```
  
- `num_residual_groups` (integer): Number of residual groups in RCAN
  - Default: 5, Range: >=1
    ```javascript
    "num_residual_groups": 5
    ```
  
- `channel_reduction` (integer): Channel reduction ratio for channel attention (no need to specify in `config.json`)
  - Default: 8, Range: >=1
    ```javascript
    "channel_reduction": 4
    ```

- `data_augmentation` (boolean): Enable/Disable data augmentation (rotation and flip) (no need to specify in `config.json`)
  - Default: True
    ```javascript
    "data_augmentation": False
    ```

- `intensity_threshold` (number): Threshold used to reject patches with low average intensity (no need to specify in `config.json`)
  - Default: 0.25, Range: >0.0
    ```javascript
    "intensity_threshold": 0.3
    ```

- `area_ratio_threshold` (number): Threshold used to reject patches with small areas of valid signal (no need to specify in `config.json`)
  - Default: 0.5, Range: 0.0~1.0
    ```javascript
    "area_ratio_threshold": 0.3
    ```

- `initial_learning_rate` (number): Initial learning rate
  - Default: 1e-4, Range: >= 1e-6
    ```javascript
    "initial_learning_rate": 1e-5
    ```

- `loss` (string): The objective function used for deep learning training (no need to specify in `config.json`)
  - Defaut: "mae", Options:  (1) “mae”= mean absolute error or (2) “mse”= mean squared error
    ```javascript
    "loss": "mse"
    ```

- `metrics` (array of strings): List of metrics to be evaluated during training (no need to specify in `config.json`)
  - Default: "psnr", Options: (1) “psnr”= Peak signal-to-noise ratio  and (2) “ssim”= structural similarity index measure
    ```javascript
    "metrics": ["psnr", "ssim"]
    ```

The default RCAN architecture is configured to be trained on a machine with 11GB GPU memory. If you encounter an OOM error during training, please try reducing model parameters such as `num_residual_blocks` or `num_residual_groups`. In the example [`config.json`](config.json), we reduce `num_residual_groups` to 3 to run on a 6GB GTX 1060 GPU.

The expected runtime is 5-10 min/epoch using the example [`config.json`](config.json) under a PC similar to our tested environment.

The loss values are saved in the training output folder. You can use TensorBoard to monitor the loss values. To use TensorBoard, run the following command and open [http://127.0.0.1:6006] in your browser.

```posh
tensorboard --host=127.0.0.1 --logdir=/path/to/training/dir
```



## Applying the Model
Trained 3D-RCAN models can be applied using the `apply.py` script. The script will automatically select the model with the lowest validation loss from the specified model directory.

There are two primary ways to apply a model:

### (Option 1) To a single image
To apply the trained model to an image, run:

```posh
python apply.py -m /path/to/training/output/dir -i input_raw_image.tif -o output.tif
```

### (Option 2) To a folder of images (batch mode)
You can turn on the “batch apply” mode by passing a foldery path to the “-i” argument, e.g.:

```posh
python apply.py -m /path/to/training/output/dir -i /path/to/input/image/dir -o /path/to/output/image/dir
```

When the input (specified by “-i”) is a folder, the output (“-o”) must be a folder too. The output folder is created by the script if it doesn’t exist yet.

### All options
- `-m` or `--model_dir` (string) [required]: Path of the folder that contains the deep learning model to be applied.
- `-i` or `--input` (string) [required]: Path of the input raw image or folder.
- `-o` or `--output` (string) [required]: Path of the output image or folder.
- `-g` or `--ground_truth` (string) [optional]: Path of the reference ground truth image or folder.
- `-b` or `--bpp` (int) [optional]: Bit depth of the output image (e.g., 8, 16, 32).
- `-B` or `--block_shape`(tuple_of_ints) [optional]: Dimensions (Z,Y,X) of the block used to divide an input image into small blocks that could fit the GPU memory.
- `-O` or `--block_overlap_shape`(tuple_of_ints) [optional]: The overlap sizes (Z,Y,X) between neighboring blocks.
-  `--normalize_output_range_between_zero_and_one` [optional]: Normalizes output intensity to the [0, 1] range, or to the full bit depth range (e.g., [0, 65535] for 16-bit) when combined with `-b`.
- `--rescale` [optional]: Performs affine rescaling to minimize MSE between restored and ground truth images, useful for comparisons similar to CARE methodology.
- `-f` or `--output_tiff_format`(str) [optional]: Sets the output TIFF format (e.g., “imagej” or “ome”).



## Denoising Model
1. Configure the settings file (`config_denoise.json`) to define the data locations for the training/validation sets and the initial network hyperparameters. An example configuration file is shown below.
   ```javascript
    {
      "training_data_dir": {
          "raw": "D:\\RCAN_dataset\\Denoising\\Tomm20_Mitochondria\\Training\\Raw",
          "gt": "D:\\RCAN_dataset\\Denoising\\Tomm20_Mitochondria\\Training\\GT"
      },
      "epochs": 300,
      "steps_per_epoch": 256,
      "num_channels": 32,
      "num_residual_blocks": 5,
      "num_residual_groups": 5,
      "input_shape": [8, 256, 256],
      "initial_learning_rate": 1e-4
    }
    ```

2. Training Command: 
   to train the model with given image pairs, provide the json file (`-c`), and path for the output model (`-o`).
    ```posh
    python train.py -c config_denoise.json -o "D:\\RCAN_dataset\\Denoising\\Tomm20_Mitochondria\\Training\\model"
    ```

3. Applying Command: 
   to apply the model trainined in the previous section, provide the model (`-m`), path for the input images (`-i`), and path for the output denoised images (`-o`).
   ```posh
   python apply.py -m "D:\\RCAN_dataset\\Denoising\\Tomm20_Mitochondria\\model" -i "D:\\RCAN_dataset\\Denoising\\Tomm20_Mitochondria\\Test\\raw" -o "D:\\RCAN_dataset\\Denoising\\Tomm20_Mitochondria\\Test\\Denoised" -b 16
    ```


    
## Notes:
(1) Do the following before initializing TensorFlow to limit TensorFlow to first GPU:
 
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Alternatively, you can choose which GPU to use for training/applying the model:

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
(2) You can find out which version of TensorFlow is installed via:

    pip show tensorflow

