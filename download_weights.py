import os
import subprocess
import zipfile
import argparse

WEIGHTS_URL = "https://zenodo.org/api/records/19021496/files/Dataset010_AirwaySegmentation.zip/content"
DATASET_NAME = "Dataset010_AirwaySegmentation"

def download_and_extract_weights(weights_dir):
    """Downloads and extracts model weights from Zenodo using wget."""
    target_dataset_dir = os.path.join(weights_dir, DATASET_NAME)
    
    if os.path.exists(target_dataset_dir):
        print(f"✅ Weights already exist at: {target_dataset_dir}")
        return True
        
    os.makedirs(weights_dir, exist_ok=True)
    zip_path = os.path.join(weights_dir, "weights.zip")
    
    try:
        print(f"📥 Downloading model weights to {weights_dir} via wget...")
        cmd = ["wget", "-O", zip_path, WEIGHTS_URL]
        subprocess.run(cmd, check=True)
        
        print("📦 Extracting weights...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
            
        os.remove(zip_path)
        print(f"🎉 Successfully downloaded and extracted to: {target_dataset_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ wget download failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Failed to download or extract weights: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download nnU-Net weights for Airway Segmentation")
    parser.add_argument("-d", "--dir", default="weights", help="Directory to save the weights (default: ./weights)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.dir)
    download_and_extract_weights(output_dir)

if __name__ == "__main__":
    main()