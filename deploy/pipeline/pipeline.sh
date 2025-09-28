input_dir=$1
output_dir=$2
modality=$3
roi_path=$4

case "$modality" in
    t1)
        task_id=Task110_PancreasT1MRI
        ;;
    t2)
        task_id=Task111_PancreasT2MRI
        ;;
    *)
        echo "Error: Unknown modality '$modality', please use t1 or t2"
        exit 1
        ;;
esac

echo "Starting segmentation..."
nnUNet_predict -tr nnTransUNetTrainerV2 -i ${input_dir} -o ${output_dir} -t ${task_id} -m 3d_fullres --folds 0

echo "Starting ROI extraction..."
python $(dirname "$0")/preprocessing/roi_extraction.py -i ${input_dir} -m ${output_dir} -o ${roi_path}

echo "Pipeline completed successfully!"
python $(dirname "$0")/classification/deploy.py -i ${roi_path} -m ${modality}