modality="t1"
script="./radiomics_classification_mRMR.py"
feature_path="./features/radiomics_features_${modality}.csv"
split_path="./splits_external_${modality}.json"
output_dir="/data2/pyq6817/CystX/radiomics_results/${modality}_mRMR_external"
python $script -f 5 --features-path $feature_path --split-path $split_path --output-dir $output_dir &
# python $script -f 1 --features-path $feature_path --split-path $split_path --output-dir $output_dir &
# python $script -f 2 --features-path $feature_path --split-path $split_path --output-dir $output_dir &
# python $script -f 3 --features-path $feature_path --split-path $split_path --output-dir $output_dir &

modality="t2"
script="./radiomics_classification_mRMR.py"
feature_path="./features/radiomics_features_${modality}.csv"
split_path="./splits_external_${modality}.json"
output_dir="/data2/pyq6817/CystX/radiomics_results/${modality}_mRMR_external"
python $script -f 5 --features-path $feature_path --split-path $split_path --output-dir $output_dir &
# python $script -f 1 --features-path $feature_path --split-path $split_path --output-dir $output_dir &
# python $script -f 2 --features-path $feature_path --split-path $split_path --output-dir $output_dir &
# python $script -f 3 --features-path $feature_path --split-path $split_path --output-dir $output_dir &