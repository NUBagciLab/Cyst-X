For the PanSegNet Segmentation code, please go to [PanSegNet](https://github.com/NUBagciLab/PaNSegNet).
Model weights and segmentation outputs are available at [HuggingFace](https://huggingface.co/phy710/Cyst-X/tree/main/Segmentation).

To train `Swin-UnetR` model with cross validation:

    chmod +x ./train.sh
    CUDA_VISIBLE_DEVICES=0 ./train.sh 1
    CUDA_VISIBLE_DEVICES=1 ./train.sh 2
Where, 1 and 2 after ./train.sh select T1 or T2 modality to be used.

To test `Swin-UnetR` model with cross validation and save segmentation outputs:

    python ./fold_test.py
    python ./fold_test.py --t 2
    
It will also output the results for LaTeX.
Please revise the --data-path to your data path.
