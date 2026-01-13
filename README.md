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

6. In Anaconda Promp, run:
    ```posh
    D:
    cd D:\Code\Python Code\RCAN (Suppose this your project folder)
    pip install -r requirements.txt ("requirements.txt" must be placed under the above folder)
    ```

7. In Anaconda Promp, verify the GPU setup:
    ```posh
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```
    
## Run Files:
8. Copy these files into the Pycharm project "RCAN" folder and modify it accordingly.

9. Run code
     - In Anaconda Prompt, run the Python file, e.g.:
     ```posh

     Python datagen_isotropic.py
     ```

11. Use ctrl-C in the Terminal to terminate the process.

12. Code run sequence: [datagen_isotropic.py](https://github.com/eexuesong/CARE/blob/main/datagen_isotropic.py) --> [training_3D.py](https://github.com/eexuesong/CARE/blob/main/training_3D.py) --> [prediction_isotropic_6degree.py](https://github.com/eexuesong/CARE/blob/main/prediction_isotropic_6degree.py)

## Notes:
 (1) Do the following before initializing TensorFlow to limit TensorFlow to first GPU:
 
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

 (2) You can find out which version of TensorFlow is installed via:

    pip show tensorflow

