# CS230 Final Project Code

Directory Structure:
- model_config.yaml           # Configuration file for training
- muscle_seg_addUnet_yaml.py  # Training Script for additional UNet model (Model 2)
- /inputs                     # Folder containing DICOM files of CT images
- /c2c_masks                  # Folder containing DICOM files of binary muscle masks from Comp2Comp model output
- /labels                     # Folder containing DICOM files of labeled muscle subgroup masks
- /logs                       # Folder containing output logs and callbacks from model training
- Visualize

To Run Model 2 Training in Commandline: 
- cd into directory with muscle_seg_addUnet_yaml.py
- activate conda {environment}
- python muscle_seg_addUnet_yaml.py model_config.yaml
