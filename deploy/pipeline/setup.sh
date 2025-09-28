# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set relative paths based on the script location
echo "export nnUNet_raw_data_base=\"$SCRIPT_DIR/segmentation_model/nnunet_raw_data_base\"" >> ~/.bashrc
echo "export nnUNet_preprocessed=\"$SCRIPT_DIR/segmentation_model/nnunet_preprocessed\"" >> ~/.bashrc
echo "export RESULTS_FOLDER=\"$SCRIPT_DIR/segmentation_model/nnunet_trained_models\"" >> ~/.bashrc

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install dicom2nifti
pip install medpy
pip install tqdm
pip install matplotlib
pip install importlib-resources >=5.12
git clone https://github.com/NUBagciLab/PaNSegNet.git
cd PaNSegNet/src
python setup.py install
pip install batchgenerators
pip install numpy==1.24.3
pip install 'monai[nibabel]' --no-deps
# source the bashrc file
source ~/.bashrc