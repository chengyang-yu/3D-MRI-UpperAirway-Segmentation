import os
import sys
import shutil
import argparse
import tempfile
import subprocess
import numpy as np
import urllib.request
import zipfile
import SimpleITK as sitk
import requests

WEIGHTS_URL = "https://zenodo.org/api/records/19021496/files/Dataset010_AirwaySegmentation.zip/content" 
DATASET_NAME = "Dataset010_AirwaySegmentation" 
# =================================================

def download_and_extract_weights(weights_dir):
    target_dataset_dir = os.path.join(weights_dir, DATASET_NAME)
    if os.path.exists(target_dataset_dir):
        return True
        
    os.makedirs(weights_dir, exist_ok=True)
    zip_path = os.path.join(weights_dir, "weights.zip")
    
    try:
        print("📥 Model weights not found. Downloading via wget...")
        cmd = ["wget", "-O", zip_path, WEIGHTS_URL]
        subprocess.run(cmd, check=True)
        
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"❌ Failed to download weights: {e}")
        return False

def convert_dicom_to_nifti(input_path, output_path, dicom_mode):
    try:
        if dicom_mode == "2d_series" and os.path.isdir(input_path):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(input_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            image = sitk.ReadImage(input_path)
        image = sitk.PermuteAxes(image, [2, 1, 0])
        sitk.WriteImage(image, output_path)
        return True
    except Exception as e:
        print(f"⚠️ Warning: Failed to convert {input_path} : {e}")
        return False

def prepare_inputs(input_folder, temp_in_folder, dicom_mode):
    file_mapping = {}
    if dicom_mode == "2d_series":
        has_direct_dicoms = any((f.endswith('.dcm') or f.endswith('.IMA')) and os.path.isfile(os.path.join(input_folder, f)) for f in os.listdir(input_folder))
        
        if has_direct_dicoms:
            base_name = os.path.basename(input_folder.rstrip('/\\'))
            temp_path = os.path.join(temp_in_folder, f"{base_name}_0000.nii.gz")
            if convert_dicom_to_nifti(input_folder, temp_path, dicom_mode):
                file_mapping[base_name] = base_name
                print(f"🔄 Converted SINGLE 2D DICOM series folder: {base_name}")
            return file_mapping 

    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        if os.path.isdir(item_path):
            if dicom_mode == "2d_series":
                base_name = item
                temp_path = os.path.join(temp_in_folder, f"{base_name}_0000.nii.gz")
                if convert_dicom_to_nifti(item_path, temp_path, dicom_mode):
                    file_mapping[base_name] = base_name
                    print(f"🔄 Converted 2D DICOM series folder: {item}")
            continue

        if item.endswith('.nii.gz') or item.endswith('.nii'):
            base_name = item.replace('.nii.gz', '').replace('.nii', '')
            temp_name = f"{base_name}_0000.nii.gz"
            temp_path = os.path.join(temp_in_folder, temp_name)
            shutil.copy(item_path, temp_path)
            file_mapping[base_name] = base_name
            print(f"📦 Prepared NIfTI: {item}")

        elif item.endswith('.dcm') or item.endswith('.IMA'):
            if dicom_mode == "3d":
                base_name = item.replace('.dcm', '').replace('.IMA', '')
                temp_name = f"{base_name}_0000.nii.gz"
                temp_path = os.path.join(temp_in_folder, temp_name)
                if convert_dicom_to_nifti(item_path, temp_path, dicom_mode):
                    file_mapping[base_name] = base_name
                    print(f"🔄 Converted 3D DICOM file to NIfTI: {item}")
            else:
                print(f"⚠️ Warning: Skipped single file '{item}' because dicom_mode is '2d_series'.")
                
    return file_mapping

def process_and_save_outputs(temp_out_folder, final_out_folder, file_mapping):
    """Corrects label values to match the JMRI paper and saves final masks."""
    for temp_file in os.listdir(temp_out_folder):
        if not temp_file.endswith('.nii.gz'):
            continue
            
        base_name = temp_file.replace('.nii.gz', '')
        if base_name not in file_mapping:
            continue
            
        original_name = file_mapping[base_name]
        temp_file_path = os.path.join(temp_out_folder, temp_file)
        final_file_path = os.path.join(final_out_folder, f"{original_name}_seg.nii.gz")
        
        img = sitk.ReadImage(temp_file_path)
        arr = sitk.GetArrayFromImage(img)
        
        safe_copy = np.copy(arr)
        arr[safe_copy == 1] = 2
        arr[safe_copy == 2] = 1 
        
        new_img = sitk.GetImageFromArray(arr)
        new_img.CopyInformation(img)
        
        sitk.WriteImage(new_img, final_file_path)
        print(f"✅ Saved segmentation: {original_name}_seg.nii.gz")

def main():
    parser = argparse.ArgumentParser(description="Minimal Inference Script for 3D-MRI Airway Segmentation")
    parser.add_argument("-i", "--input", required=True, help="Path to input directory (.nii.gz files or DICOM folders)")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    parser.add_argument("--dicom_mode", choices=["3d", "2d_series"], default="3d", help="Mode for DICOM processing: '3d' for single file per volume, '2d_series' for folder of slices.")
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input)
    output_folder = os.path.abspath(args.output)

    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder not found: {input_folder}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)

    print("\n🚀 Starting Automated Inference Pipeline...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights")
    
    if not download_and_extract_weights(weights_dir):
        sys.exit(1)
        
    os.environ["nnUNet_results"] = weights_dir
    dataset_id = DATASET_NAME.split("_")[0].replace("Dataset", "")

    with tempfile.TemporaryDirectory() as temp_in, tempfile.TemporaryDirectory() as temp_out:
        print("\n🔍 Step 1: Preparing and formatting input files...")
        file_mapping = prepare_inputs(input_folder, temp_in, dicom_mode=args.dicom_mode)
        
        if not file_mapping:
            print("❌ Error: No valid NIfTI files or DICOM folders found.")
            sys.exit(1)

        print("\n🤖 Step 2: Running nnU-Net prediction...")
        cmd = [
            "nnUNetv2_predict",
            "-i", temp_in,
            "-o", temp_out,
            "-d", dataset_id,
            "-c", "3d_fullres",
            "-f", "0"
        ]
        
        try:
            subprocess.run(cmd, env=os.environ.copy(), check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ nnU-Net Execution Failed with error code {e.returncode}")
            sys.exit(1)

        print("\n🔄 Step 3: Saving...")
        process_and_save_outputs(temp_out, output_folder, file_mapping)

    print(f"\n🎉 All tasks completed! Final results are located in: {output_folder}")

if __name__ == "__main__":
    main()