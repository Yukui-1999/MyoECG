
Batch_size=256
Seed=42
Lr=5e-5
Input_modality='ECG'
PYTHON=${PYTHON:-"python"} 
Dis='cm'
Threshold_method='youden'
Ecg_config_path='st_mem_align.yaml'
ECG_type=$(echo "$Ecg_config_path" | awk -F'/' '{print $NF}' | sed 's/\(.yaml\)//')
NOW=$(date +"%Y%m%d_%H%M%S")
CKPT_DIR=SimpleTest/MIMIC_test_output



export CUDA_VISIBLE_DEVICES=1
${PYTHON} -u main_SimpleTest.py \
    --input_modality ${Input_modality} \
    --batch_size ${Batch_size} \
    --seed ${Seed} \
    --blr ${Lr} \
    --ecg_config_path ${Ecg_config_path} \
    --output_dir ${CKPT_DIR} \
    --ecg_model 'stmem' \
    --num_classes 1 \
    --dis ${Dis} \
    --threshold_method ${Threshold_method} \
    --test_dir_name 'test_infer_mimic' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log