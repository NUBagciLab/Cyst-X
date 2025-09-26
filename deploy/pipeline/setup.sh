# Ask user for raw data path
read -p $"Where do you want to put your raw data? For example, '/data2/pyq6817/nnUNetv1/nnUNet_raw_data_base' -->" raw_data_path
export nnUNet_raw_data_base="$raw_data_path"

# Ask user for preprocessed data path
read -p $"Where do you want to put your preprocessed data? For example, '/data2/pyq6817/nnUNetv1/nnUNet_preprocessed' -->" preprocessed_path

export nnUNet_preprocessed="$preprocessed_path"

# Ask user for trained models path
read -p $"Where do you want to put your results/trained models? For example, '/data2/pyq6817/nnUNetv1/nnUNet_trained_models' -->" results_path
export RESULTS_FOLDER="$results_path"

echo "Paths set:"
echo "nnUNet_raw_data_base=$nnUNet_raw_data_base"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "RESULTS_FOLDER=$RESULTS_FOLDER"

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/NUBagciLab/PaNSegNet.git
cd PaNSegNet/src
python setup.py
pip install 'monai[nibabel]'
