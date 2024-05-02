# UC Berkeley - UCSF FAST Collaboration

## Codebase Structure Overview

This is a codebase for the UCSF FAST project.The following is a brief overview of the important components:

- **common**: Common utility functions and classes.
  - **datasets/\_\_init__.py**: Custom dataset classes for handling data.
  - **logging/\_\_init__.py**: 
  - **models**: Custom models for training.
    - **large**:
    - **small**:
- **experiments**: Main directory for experimental scripts.
  - **utils**: Contains utility scripts such as `accessing_mdai_data.py` to facilitate data operations.
  - **train.py**: Main training script for models.
  - **accessing_mdai_data.py**: Script for accessing MDai annotations. 
- **figs**: Directory for storing figures and plots generated from scripts or notebooks.
- **notebooks**: Contains Jupyter notebooks for various purposes such as data exploration and visualization.
  - **create_3D_image.ipynb**: Notebook for creating 3D images of clinical masks.
  - **segmentation_visualizer_test.ipynb**: Notebook for testing segmentation visualization.
  - **view_images.ipynb**: Simple notebook to view images.
- **scripts**: Directory for Slurm scripts for running experiments on the cluster.
- **config.yaml**: Configuration file to manage settings and configurations of experiments.

## Usage

To use the codebase, follow these steps:
1. Clone the repository.
2. Install the required packages using config.yaml: `pip install -r config.yaml`.
3. Download the data from the Kornblith Lab folder on UCSF Research Analysis Environment (RAE). In particular, download
the directories:
    - `M:\TempAnnotations_Morison\DCMFRM\DCMFRM`
    - `M:\Positive_13_15\P1315_Images`
    - `M:\Positive_13_15\InstanceRecord.csv`
and update the `mdai_args` dictionary in `experiments/accessing_mdai_data.py` with the appropriate paths.
4. Run the `experiments/accessing_mdai_data.py` script to fetch the annotations from MDai and generate the necessary data files.
5. ...
6. Profit.