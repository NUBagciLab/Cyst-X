pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
git checkout nnunetv1
pip install -e .

export nnUNet_raw_data_base="/data2/pyq6817/nnUNetv1/nnUNet_raw_data_base" 
export nnUNet_preprocessed="/data2/pyq6817/nnUNetv1/nnUNet_preprocessed" 
export RESULTS_FOLDER="/data2/pyq6817/nnUNetv1/nnUNet_trained_models"  
pip install 'monai[nibabel]'
