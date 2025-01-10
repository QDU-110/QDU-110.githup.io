# QDU-110.githup.io
Neuroimaging Synthesis and Classification for Alzheimer’s Disease (NSC-AD-TIDL-SHAP)

Introduction

This repository contains the implementation of a task-integrated deep learning framework for neuroimaging synthesis and Alzheimer’s disease (AD) classification. The framework combines 3D BicycleGAN, Multimodality Fusion Neural Network (MFNet), and SHAP Analysis to enhance predictive performance, interpretability, and diagnostic accuracy. By utilizing multimodal data (MRI, PET, and clinical data), this approach facilitates improved understanding, diagnosis, and treatment of Alzheimer’s disease.

Key highlights of the framework include:

Neuroimaging synthesis to generate high-quality PET scans from MRI data using 3D BicycleGAN.
Disease classification with a fusion of multimodal data (neuroimaging and clinical data).
SHAP Analysis for interpretability of the classification model and identification of important features.
This framework is validated using the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset and employs Voxel-Based Morphometry (VBM) for biological plausibility of the findings


Code Explanation

QDU-110.githup.io/
├── image_quality_calculate/   # Image quality evaluation metrics (includes MAE, MSE, SSIM, PSNR, etc.)
├── models/                    # Model definitions (e.g., DenseUNet, GAN models)
├── options/                   # Configuration files (hyperparameter settings)
├── pretrained_models/         # Pretrained model files
├── scripts/                   # Main training, testing, and evaluation scripts
├── util/                      # Utility functions
├── checkpoints/               # Saved model checkpoint files
├── ceshi_111.py               # Example test script (might need to be moved into the scripts folder)
├── DenseUNet.py               # DenseUNet model definition
├── loss_ceshi.py              # Loss function test script
├── error_map.py               # Error map generation script
├── MAE.py                     # Mean Absolute Error (MAE) calculation
├── MMD.py                     # Maximum Mean Discrepancy (MMD) calculation
├── MSE.py                     # Mean Squared Error (MSE) calculation
├── PSNR.py                    # Peak Signal-to-Noise Ratio (PSNR) calculation
├── SSIM.py                    # Structural Similarity Index (SSIM) calculation
├── quality_calculate.py       # Entry point for image quality evaluation
├── model_ceshi.py             # Model testing script
├── test.py                    # Main testing script
├── train.py                   # Main training script
└── README.md                  # Project documentation

Need to download the required software in the software file.




