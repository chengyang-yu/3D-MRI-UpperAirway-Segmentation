# 3D MRI Upper Airway Segmentation 

This repository provides the official inference pipeline for the upper airway segmentation model described in our JMRI 2026 paper: **"Analysis of Upper Airway Morphology Using Four-Dimensional Dynamic MRI With Active Deep Learning-Based Automatic Segmentation"** (Yu CY et al.).

The model is built upon the nnU-Net framework and has been trained for accurate segmentation of the upper airway and epiglottic airway from 3D MRI scans. For the precise anatomical boundaries and definitions of these regions, please refer to our paper.

## 1. Clone & Installation

First, clone this repository to your local machine and navigate into the directory:

```bash
git clone https://github.com/chengyang-yu/3D-MRI-UpperAirway-Segmentation
cd 3D-MRI-UpperAirway-Segmentation
```

We recommend using Conda to create a clean environment:

```bash
conda create -n airwayseg_env python=3.10 -y
conda activate airwayseg_env
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. How to Run Inference

You do not need to manually download the model weights to start. Our inference script (`predict.py`) will automatically check, download, and extract the required weights from Zenodo upon its first run. 

Our script supports both single 3D files (NIfTI, DICOM) and 2D DICOM series folders.

### Option A: 3D Mode (Default)
Use this mode if your input directory contains single 3D files (e.g., `.nii.gz`, `.nii`, or single 3D `.dcm` / `.IMA` files).

```bash
python3 predict.py -i /path/to/input_folder -o /path/to/output_folder
```

### Option B: 2D DICOM Series Mode
Use this mode for 2D DICOM slices. Our script automatically detects your folder structure and supports two scenarios:
* **Single Scan**: The input directory directly contains the 2D slices (e.g., `001.dcm`, `002.dcm`).
* **Batch Processing**: The input directory contains multiple sub-folders, where each sub-folder holds the slices for a specific patient/scan.

In both cases, the script will automatically reorient and stack the slices into 3D volumes.

```bash
python3 predict.py -i /path/to/input_folder -o /path/to/output_folder --dicom_mode 2d_series
```

## 3. Download Model Weights (For Custom Usage)

If you wish to bypass our inference script and use the raw nnU-Net commands or integrate the model into your own custom pipeline, you can manually download the weights using the script:

```bash
python3 download_weights.py
```
*(By default, this will create a `weights` folder in your current directory. You can also specify a custom path using `-d /your/custom/path`)*

## 4. ⚠️ Important Note on Data Format (Orientation)

If you are providing **NIfTI (.nii.gz)** files directly, please ensure the spatial dimensions are ordered as **(Z, Y, X)**. Specifically:
* **Z**: Cranial-caudal
* **Y**: Ventral-dorsal
* **X**: Left-right

If your data is currently in (X, Y, Z) format, you must permute the axes before running inference. Alternatively, you can provide raw DICOM files, and our script will automatically handle the spatial orientation and conversion for you.

## 5. Label Definitions
The output segmentation masks will contain the following label values:
* **Label 0**: Background
* **Label 1**: Upper Airway
* **Label 2**: Epiglottic Airway

## Citation
If you find this tool or the model weights useful in your research, please cite our paper:
> Yu CY, et al. "Analysis of Upper Airway Morphology Using Four-Dimensional Dynamic MRI With Active Deep Learning-Based Automatic Segmentation." Journal of Magnetic Resonance Imaging (JMRI), 2026.
