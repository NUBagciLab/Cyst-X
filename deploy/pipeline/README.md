Setting up for segmentation and classification:
On Linux: 

    bash setup.sh

Few parameters are needed for prediction and segmentation,1-input_folder: where you put your images, 2-output folder: where the segmentation mask goes, 3-modality: t1 or t2, 4-ROI folder: where the region of interest goes,for example:

    bash '/pipeline/pipeline.sh' '/pipeline/example/test_input' '/pipeline/example/test_segmentation' t1 '/pipeline/example/test_preprocessed'

