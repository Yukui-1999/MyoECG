
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
    --test_dir_name 'test_infer_mimic_MyoECG' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log




Batch_size=256
Seed=42
Lr=5e-5
Input_modality='ECG'
PYTHON=${PYTHON:-"python"} 
Dis='cm'
Threshold_method='youden'
Ecg_config_path='st_mem.yaml'
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
    --test_dir_name 'test_infer_mimic_ST-MEM' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log




Batch_size=32
Seed=42
Lr=5e-5
Input_modality='ECG'
PYTHON=${PYTHON:-"python"} 
Dis='cm'
Threshold_method='youden'
ECG_model='ecg_found'

NOW=$(date +"%Y%m%d_%H%M%S")
CKPT_DIR=SimpleTest/MIMIC_test_output


export CUDA_VISIBLE_DEVICES=0
${PYTHON} -u main_SimpleTest.py \
    --input_modality ${Input_modality} \
    --batch_size 256 \
    --seed ${Seed} \
    --blr ${Lr} \
    --output_dir ${CKPT_DIR} \
    --ecg_model ${ECG_model} \
    --num_classes 1 \
    --dis ${Dis} \
    --threshold_method ${Threshold_method} \
    --test_dir_name 'test_infer_mimic_ECGFounder' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log




Batch_size=32
Seed=42
Lr=5e-5
Input_modality='ECG'
PYTHON=${PYTHON:-"python"} 
Dis='cm'
Threshold_method='youden'
ECG_model='ecgfm_ked'

NOW=$(date +"%Y%m%d_%H%M%S")
CKPT_DIR=SimpleTest/MIMIC_test_output


export CUDA_VISIBLE_DEVICES=3
${PYTHON} -u main_SimpleTest.py \
    --input_modality ${Input_modality} \
    --batch_size 256 \
    --seed ${Seed} \
    --blr ${Lr} \
    --output_dir ${CKPT_DIR} \
    --ecg_model ${ECG_model} \
    --num_classes 1 \
    --dis ${Dis} \
    --threshold_method ${Threshold_method} \
    --test_dir_name 'test_infer_mimic_ECGFM-KED' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log



Batch_size=32
Seed=42
Lr=5e-5
Input_modality='ECG'
PYTHON=${PYTHON:-"python"} 
Dis='cm'
Threshold_method='youden'
ECG_model='fg_clep'

NOW=$(date +"%Y%m%d_%H%M%S")
CKPT_DIR=SimpleTest/MIMIC_test_output


export CUDA_VISIBLE_DEVICES=3
${PYTHON} -u main_SimpleTest.py \
    --input_modality ${Input_modality} \
    --batch_size 256 \
    --seed ${Seed} \
    --blr ${Lr} \
    --output_dir ${CKPT_DIR} \
    --ecg_model ${ECG_model} \
    --num_classes 1 \
    --dis ${Dis} \
    --threshold_method ${Threshold_method} \
    --test_dir_name 'test_infer_mimic_FG-CLEP' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log




Batch_size=32
Seed=42
Lr=5e-5
Input_modality='ECG'
PYTHON=${PYTHON:-"python"} 
Dis='cm'
Threshold_method='youden'
ECG_model='merl'

NOW=$(date +"%Y%m%d_%H%M%S")
CKPT_DIR=SimpleTest/MIMIC_test_output


export CUDA_VISIBLE_DEVICES=3
${PYTHON} -u main_SimpleTest.py \
    --input_modality ${Input_modality} \
    --batch_size 256 \
    --seed ${Seed} \
    --blr ${Lr} \
    --output_dir ${CKPT_DIR} \
    --ecg_model ${ECG_model} \
    --num_classes 1 \
    --dis ${Dis} \
    --threshold_method ${Threshold_method} \
    --test_dir_name 'test_infer_mimic_MERL' \
    --only_test \
    ${PY_ARGS} \
    2>&1 | tee -a ${CKPT_DIR}/train-${NOW}.log