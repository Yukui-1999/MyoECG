# MyoECG

Code and resources for **Cardiac MRI-informed electrocardiography enables biologically grounded cardiomyopathy detection**.

MyoECG transfers information from cardiac magnetic resonance (CMR) to an ECG encoder through two stages of ECG–CMR pretraining. The resulting ECG model can be fine-tuned for cardiomyopathy detection. CMR is used during pretraining; the example inference pipeline requires ECG only.

![MyoECG overview](Figure1.png)

## Resource archive

Model weights and example data are being deposited on Hugging Face at **[10.57967/hf/10893](https://huggingface.co/Yukui1999/MyoECG)**. The upload is in progress while this README is prepared. Download availability depends on completion and publication of that record. Use the published record for the final file inventory and citation, and retain the specific record used in your experiments.

The resource package is organized as follows. Filenames are case-sensitive on Linux.

```text
ModelWeights/
├── AfterFinetuned_CMdetectModel/
│   ├── best-auc-MyoECG.pth
│   ├── best-auc-ST-MEM.pth
│   ├── best-auc-ECGFounder.pth
│   ├── best-auc-ECGFM-KED.pth
│   ├── best-auc-FG-CLEP.pth
│   └── best-auc-MERL.pth
├── ECGBaselineModels_OriginalGithub/
│   ├── STMEM/                 # Original weights and github.txt
│   ├── ECGFounder/
│   ├── ECGFM-KED/
│   ├── FG_CLEP/
│   └── MERL/
├── PretrainedCMRmodel/
│   ├── CMR_PretrainedModel.pth
│   └── Cine_SAX_Exploratory_Analysis/
│       ├── SaxCMR_PretrainedModel.pth
│       └── ECGPretrainedModel_with_SaxCMR.pth
├── AfterTwoStageECGCMRPretrained_ECGModel.pth
└── Simple_data_mimic.pkl
```

| Resource | Purpose |
| --- | --- |
| `AfterTwoStageECGCMRPretrained_ECGModel.pth` | MyoECG ECG checkpoint after two-stage ECG–CMR pretraining, before task-specific cardiomyopathy fine-tuning. |
| `AfterFinetuned_CMdetectModel/best-auc-*.pth` | Six models fine-tuned using FCCCH cardiomyopathy labels: MyoECG and five comparators. These are the detection checkpoints used by SimpleTest. |
| `ECGBaselineModels_OriginalGithub/` | Original comparator checkpoints and explicit links to their public implementations. |
| `PretrainedCMRmodel/CMR_PretrainedModel.pth` | Pretrained long-axis CMR MAE checkpoint used to initialize the CMR encoder. |
| `PretrainedCMRmodel/Cine_SAX_Exploratory_Analysis/` | Additional short-axis CMR pretraining and short-axis-aligned ECG checkpoints for the exploratory analysis. |
| `Simple_data_mimic.pkl` | Example preprocessed MIMIC ECG data for the SimpleTest pipeline. |

The pretrained ECG checkpoint and the fine-tuned MyoECG detection checkpoint serve different purposes. Use `best-auc-MyoECG.pth` for the binary detection example below.

## Code organization

| Location | Contents |
| --- | --- |
| `main_SimpleTest.py` | ECG-only inference with six fine-tuned models. |
| `scripts/simple_test_mimic.sh` | Original six-model example launcher. |
| `st_mem_align.yaml`, `st_mem.yaml` | MyoECG and ST-MEM configurations for SimpleTest. |
| `data/dataset.py` | Cohort loaders and model-specific ECG transformations. |
| `engine_Caldownstream.py` | Binary training and inference routines. |
| `util/Val_class.py`, `util/val_result.py` | Threshold selection, metrics, confidence intervals and result export. |
| `ECG_baselines/` | Comparator implementations included with this release. |
| `Align_Code/script/` | Primary semantic-teacher and two-stage ECG–CMR training launchers. |
| `Align_Code/SaCMR_encoder/` | Exploratory short-axis CMR MAE pretraining. |
| `Align_Code/SaCMR_ECG_Align/` | Exploratory short-axis CMR teacher and ECG alignment. |
| `main_Cla*.py` | Cohort-specific downstream training, inference and analysis. |

## Environment setup

Use Linux with Bash and an NVIDIA GPU for the documented inference and training commands. The original dependency export records PyTorch `1.12.0+cu113` and torchvision `0.13.0+cu113`. The portable environment specification preserves these CUDA 11.3 wheel versions.

`environment.yml` is a reconstruction of the supplied dependencies, **not yet a fresh export or a clean-install validation of the authors' working Linux environment**. Python 3.8 is the provisional target inferred from the archived code environment and must be checked against the working server before final release.

From the repository root:

```bash
conda env create -f environment.yml
conda activate myoecg
python -m pip check
```

The dependency files have separate scopes:

- `requirements.txt` contains core MyoECG/ST-MEM, evaluation and alignment dependencies.
- `requirements-baselines.txt` includes the core file plus imports required by the six-model example. It is installed by `environment.yml`.
- The individual baseline repositories retain their own requirements for their independent training workflows. Use separate environments for those workflows; do not combine all their dependency files into the example environment.

Local `file://` build paths and unrelated packages from the original full environment export have been removed. Known package versions have been retained. The additional `fastai==1.0.61` requirement comes from the included ECGFM-KED requirements; its exact version on the working server still needs confirmation. A compatible NVIDIA driver is required. CPU-only execution of the preserved SimpleTest program is not established because several checkpoint loads do not specify a CPU device mapping.

## Five comparator models

The implementations are included under `ECG_baselines/`, except ST-MEM, whose encoder is under `models/`. Original checkpoints are stored separately in the Hugging Face resource package. Each baseline resource folder contains a `github.txt` with the source URL.

| Model | Public implementation | Code in this repository | Original checkpoint under `ECGBaselineModels_OriginalGithub/` |
| --- | --- | --- | --- |
| ST-MEM | [vuno/ST-MEM](https://github.com/vuno/ST-MEM) | `models/encoder/st_mem_vit.py` | `STMEM/st_mem_vit_base_encoder.pth` |
| ECGFounder | [PKUDigitalHealth/ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) | `ECG_baselines/ECGFounder/` | `ECGFounder/12_lead_ECGFounder.pth` |
| ECGFM-KED | [control-spiderman/ECGFM-KED](https://github.com/control-spiderman/ECGFM-KED) | `ECG_baselines/ECGFM_KED/` | `ECGFM-KED/best_valid_all_increase_with_augment_epoch_3.pt` |
| FG-CLEP | [CuCl-2/FG-CLEP](https://github.com/CuCl-2/FG-CLEP) | `ECG_baselines/FG_CLEP/` | `FG_CLEP/pytorch_model.bin` |
| MERL | [cheliu-computation/MERL-ICML2024](https://github.com/cheliu-computation/MERL-ICML2024) | `ECG_baselines/MERL_ICML2024/` | `MERL/best_encoder.pth` |

The six `best-auc-*.pth` files are the FCCCH-fine-tuned comparison models, rather than the original upstream pretrained checkpoints. Consult the upstream repositories for original model papers and licenses; their license files are retained where included.

## Example inference pipeline

### Arrange the downloads

The following commands assume the code directory and extracted `ModelWeights` directory are siblings:

```text
workspace/
├── ECG_CMR_CMAI_github1013_upload/
└── ModelWeights/
```

Start in the code directory. Copy the example data and all six fine-tuned checkpoints:

```bash
WEIGHTS=../ModelWeights
mkdir -p SimpleTest/MIMIC_test_data SimpleTest/MIMIC_test_output
cp "$WEIGHTS/Simple_data_mimic.pkl" SimpleTest/MIMIC_test_data/Simple_data_mimic.pkl
cp "$WEIGHTS"/AfterFinetuned_CMdetectModel/best-auc-*.pth SimpleTest/MIMIC_test_output/
```

The preserved ECGFounder and MERL loaders initialize their original models before applying the fine-tuned checkpoint, so also copy these two files:

```bash
mkdir -p ECG_baselines/ECGFounder/checkpoint ECG_baselines/MERL_ICML2024
cp "$WEIGHTS/ECGBaselineModels_OriginalGithub/ECGFounder/12_lead_ECGFounder.pth" \
  ECG_baselines/ECGFounder/checkpoint/12_lead_ECGFounder.pth
cp "$WEIGHTS/ECGBaselineModels_OriginalGithub/MERL/best_encoder.pth" \
  ECG_baselines/MERL_ICML2024/best_encoder.pth
```

The expected example layout is:

```text
SimpleTest/
├── MIMIC_test_data/
│   └── Simple_data_mimic.pkl
└── MIMIC_test_output/
    ├── best-auc-MyoECG.pth
    ├── best-auc-ST-MEM.pth
    ├── best-auc-ECGFounder.pth
    ├── best-auc-ECGFM-KED.pth
    ├── best-auc-FG-CLEP.pth
    └── best-auc-MERL.pth
```

CMR weights, the two-stage ECG pretraining checkpoint and the remaining original baseline checkpoints are not required for this example inference path. CMR images are not loaded during SimpleTest inference.

### Run MyoECG on one GPU

Run from the repository root with the Conda environment activated:

```bash
CUDA_VISIBLE_DEVICES=0 python main_SimpleTest.py \
  --device cuda:0 \
  --input_modality ECG \
  --ecg_model stmem \
  --ecg_config_path st_mem_align.yaml \
  --output_dir SimpleTest/MIMIC_test_output \
  --test_dir_name test_infer_mimic_MyoECG \
  --num_classes 1 --dis cm \
  --batch_size 32 --num_workers 0 --seed 42 \
  --only_test
```

`CUDA_VISIBLE_DEVICES=0` selects physical GPU 0; `cuda:0` is the selected GPU's index inside the process. Lower the batch size if GPU memory is insufficient. The configurations are example-specific and point to the corresponding fine-tuned files under `SimpleTest/MIMIC_test_output/`.

### Evaluate all six models

This loop uses a single selected GPU and the existing Python entry point. The import-path setting enables FG-CLEP's top-level `clep` import without changing the preserved example implementation.

```bash
export PYTHONPATH="$PWD/ECG_baselines/FG_CLEP${PYTHONPATH:+:$PYTHONPATH}"

for MODEL in MyoECG ST-MEM ECGFounder ECGFM-KED FG-CLEP MERL; do
  case "$MODEL" in
    MyoECG)     ARCH=stmem;     CONFIG=st_mem_align.yaml ;;
    ST-MEM)     ARCH=stmem;     CONFIG=st_mem.yaml ;;
    ECGFounder) ARCH=ecg_found; CONFIG=st_mem.yaml ;;
    ECGFM-KED)  ARCH=ecgfm_ked; CONFIG=st_mem.yaml ;;
    FG-CLEP)    ARCH=fg_clep;   CONFIG=st_mem.yaml ;;
    MERL)       ARCH=merl;      CONFIG=st_mem.yaml ;;
  esac
  CUDA_VISIBLE_DEVICES=0 python main_SimpleTest.py \
    --device cuda:0 --input_modality ECG \
    --ecg_model "$ARCH" --ecg_config_path "$CONFIG" \
    --output_dir SimpleTest/MIMIC_test_output \
    --test_dir_name "test_infer_mimic_${MODEL}" \
    --num_classes 1 --dis cm \
    --batch_size 32 --num_workers 0 --seed 42 --only_test || break
done
```

The configuration argument is used only by the `stmem` architecture. The other branches select their model directly. Keep the `test_dir_name` values exactly as shown: the entry point uses them to select the matching fine-tuned checkpoint.

The original `bash scripts/simple_test_mimic.sh` launcher is also retained. It selects physical GPU indices 1, 1, 0, 3, 3 and 3 for the six runs. Use it on a matching multi-GPU setup; use the commands above for a single-GPU setup. Set the same FG-CLEP `PYTHONPATH` before using that launcher. The example entry point, YAML configurations, original launcher and shared inference behavior have not been changed.

### Inputs and preprocessing

The example pickle is read through `ECGBaseMIMICDis_SimpleTest` in `data/dataset.py`, using its `samples` and `targets` entries. The MIMIC transformations expect 12-lead ECG signals sampled at 500 Hz; the typical ten-second input contains 5,000 samples per lead. The loader applies model-specific processing:

| Model | Inference transformation |
| --- | --- |
| MyoECG and ST-MEM | Resample to 250 Hz; take three 2,250-sample crops; apply 0.67 Hz high-pass and 40 Hz low-pass filtering; standardize each crop. Average crop logits before sigmoid. |
| ECGFounder | Handle NaNs, reorder the relevant augmented limb leads, apply notch/band-pass and baseline correction, then standardize. |
| ECGFM-KED | Resample to 100 Hz and standardize. |
| FG-CLEP | Retain 500 Hz and scale each lead to the range from −1 to 1. |
| MERL | Retain 500 Hz and scale jointly across leads and time to the range from −1 to 1. |

Consult the actual transformation definitions before preparing another dataset. Do not apply the same processing a second time to already transformed tensors.

## Evaluation procedures and outputs

SimpleTest loads the model parameters from `checkpoint['model']` and the operating threshold from `checkpoint['metrics']['optimal_threshold']`. It **does not optimize the threshold on the example test labels**. Passing `--threshold_method youden` does not reselect the threshold during this inference path.

For each model, results are written to:

```text
SimpleTest/MIMIC_test_output/test_infer_mimic_<MODEL>/
├── metrics.csv
└── y_true_pred.npz
```

`y_true_pred.npz` contains `y_true` and `y_pred`. `metrics.csv` reports sensitivity, specificity, accuracy, AUROC (`AUC`), positive predictive value, negative predictive value, F1 score and Cohen's kappa. The binary metric helper uses 1,000 stratified bootstrap resamples, seed 42 and percentile 95% confidence intervals, keeping the checkpoint threshold fixed. CSV thresholded metrics use `probability >= threshold`; console metrics use `probability > threshold`, which can differ when a prediction equals the threshold.

To recompute the MyoECG CSV metrics from saved predictions without running the model again:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
import torch
from util.Val_class import calculate_metrics
from util.val_result import save_metrics_to_csv

out = Path('SimpleTest/MIMIC_test_output')
result_dir = out / 'test_infer_mimic_MyoECG'
with np.load(result_dir / 'y_true_pred.npz', allow_pickle=False) as saved:
    y_true, y_pred = saved['y_true'], saved['y_pred']
checkpoint = torch.load(out / 'best-auc-MyoECG.pth', map_location='cpu')
threshold = checkpoint['metrics']['optimal_threshold']
metrics = calculate_metrics(y_true, y_pred, threshold=threshold,
                            n_bootstrap=1000, seed=42, stratified=True)
save_metrics_to_csv(metrics, result_dir / 'metrics_recomputed.csv')
PY
```

The shared output directory also receives `log.txt` and TensorBoard events. The original shell launcher additionally saves timestamped console logs. Re-running a model with the same result directory replaces its per-model result files; retain a copy when comparing runs.

The example is an approximately balanced MIMIC subset intended to demonstrate loading, preprocessing, inference and metric calculation. It does not reproduce the full MIMIC cohort or all study datasets. In particular, prevalence-dependent metrics such as PPV and NPV should not be interpreted as estimates for the full clinical cohort. Any result files already included in the code package are reference outputs, not results of a new execution on your machine.

For downstream evaluation on your own authorized data, use the corresponding `main_Cla*.py` entry point, provide the expected cohort tables and model configuration, and use validation-derived thresholds on held-out test data. Binary and three-class routines are separate; the six supplied `AfterFinetuned_CMdetectModel` files are the binary cardiomyopathy detection models.

## Primary pretraining scripts in Align_Code/script

These three launchers implement semantic-teacher training followed by two stages of ECG alignment:

| Order | Launcher | Role and output |
| --- | --- | --- |
| 1 | `Align_Code/script/train_cmr_sem_teacher.sh` | Initialize the long-axis CMR encoder from MAE weights and train a semantic/phenotype teacher. Produces `best-cmr-pheno.pth`. |
| 2 | `Align_Code/script/stmem_sem_stage1.sh` | Align the ECG encoder to the frozen CMR teacher using paired UKB data. Produces `best-loss.pth` and `st_mem_aligned_ecg_sem_stage1.yaml`. |
| 3 | `Align_Code/script/stmem_sem_stage2.sh` | Adapt the ECG encoder using paired FCCCH/Henan data and the frozen teacher. Produces `best-loss.pth` and `st_mem_aligned_ecg_sem_stage2.yaml`. |

The distributed `CMR_PretrainedModel.pth` is the **CMR encoder initialization checkpoint**. It is not interchangeable with the full `best-cmr-pheno.pth` teacher checkpoint, which also contains learned semantic and phenotype heads. Generate that teacher with step 1 before steps 2 and 3. The archived `Align_Code/CMR_encoder/main_pretrain.py` references an external `mae` package that is not included; it is not a self-contained command for repeating the original long-axis MAE training.

### Prepare initialization weights and cohort tables

From the repository root:

```bash
WEIGHTS=../ModelWeights
mkdir -p Align_Code/checkpoints/cmr Align_Code/checkpoints/st_mem checkpoints/ecg
cp "$WEIGHTS/PretrainedCMRmodel/CMR_PretrainedModel.pth" \
  Align_Code/checkpoints/cmr/CMR_PretrainedModel.pth
cp "$WEIGHTS/ECGBaselineModels_OriginalGithub/STMEM/st_mem_vit_base_encoder.pth" \
  Align_Code/checkpoints/st_mem/st_mem_vit_base_encoder.pth
cp "$WEIGHTS/AfterTwoStageECGCMRPretrained_ECGModel.pth" \
  checkpoints/ecg/AfterTwoStageECGCMRPretrained_ECGModel.pth
```

The last file is an already pretrained ECG model for subsequent fine-tuning; it is not the initialization used to repeat both alignment stages from ST-MEM.

The primary training workflow expects local authorized data and partition tables, including:

- `Align_Code/data/UKB/split/train_data.csv`, `val_data.csv`, `test_data.csv`.
- UKB table columns `Eid`, `ecg_list`, `sa_cmr`, `la_cmr`, `sex`, `age`, `BMI`, plus the phenotype columns listed in `Align_Code/util/losses_pheno.py`.
- `Align_Code/data/HenanData/seed_3407/train.xlsx`, `valid.xlsx`, `test.xlsx` for the FCCCH/Henan stage.

The UKB loader currently reads both short-axis and long-axis CMR image and segmentation tensors, even when the teacher uses long-axis CMR. Providing only ECG files or only one CMR view does not satisfy that loader. Resource paths in cohort tables must match the loader's expected format. UKB legacy-loader ECG paths can be relative to `Align_Code/data/UKBData/processed_ecg/`, and CMR paths to `Align_Code/data/UKBData/cmr/`. See `Align_Code/data/dataset.py` for the complete schema and preprocessing.

### Run teacher training and the two alignment stages

Run these commands from the repository root. Each launcher locates and enters `Align_Code` automatically. **Relative resource arguments below are therefore relative to `Align_Code`, not to the repository root.** Use a distinct run label in place of `example` and carry that label through all three commands.

```bash
# Step 1: train the semantic/phenotype CMR teacher.
GPU=0 DEVICE=cuda:0 NOW=example \
PY_ARGS='--Lax_cmr_pretrained_weights checkpoints/cmr/CMR_PretrainedModel.pth' \
bash Align_Code/script/train_cmr_sem_teacher.sh

# Step 2: UKB ECG alignment with the frozen teacher.
GPU=0 DEVICE=cuda:0 NOW=example \
CMR_TEACHER_PATH='outputs/ECG_CMR/cmr_sem_teacher_example_bs128_lr0.001_seed42/best-cmr-pheno.pth' \
bash Align_Code/script/stmem_sem_stage1.sh

# Step 3: FCCCH/Henan alignment initialized from stage 1.
GPU=0 DEVICE=cuda:0 NOW=example \
CMR_TEACHER_PATH='outputs/ECG_CMR/cmr_sem_teacher_example_bs128_lr0.001_seed42/best-cmr-pheno.pth' \
ECG_SEM_STAGE1_CONFIG_PATH='outputs/ECG_CMR/ecg_sem_stage1UKB_example_bs256_lr0.001_seed42/st_mem_aligned_ecg_sem_stage1.yaml' \
bash Align_Code/script/stmem_sem_stage2.sh
```

With these default training settings, stage 2 saves its checkpoint and downstream YAML under `Align_Code/outputs/ECG_CMR/ecg_sem_stage2FCCCH_example_bs64_lr0.0001_seed42/`. If batch size, learning rate, seed or run label changes, use the actual output paths printed by the preceding stage.

The primary launchers accept environment overrides `PYTHON`, `GPU`, `DEVICE`, `NOW` and `PROJECT_DIR`. Additional Python arguments are supplied through `PY_ARGS`; arguments appended after these three shell filenames are not forwarded. For direct Python invocation, first enter `Align_Code` and inspect the corresponding entry point's `--help`.

## Exploratory short-axis CMR analysis

The resource archive additionally contains:

- `SaxCMR_PretrainedModel.pth`: the short-axis CMR MAE checkpoint.
- `ECGPretrainedModel_with_SaxCMR.pth`: the ECG checkpoint after short-axis alignment stage 2. It is a pretrained ECG checkpoint, not one of the six FCCCH binary detection checkpoints.

The short-axis implementation is separate from the primary `Align_Code/script/` workflow. `Align_Code/SaCMR_encoder/scripts/pretrain.sh` launches short-axis MAE pretraining. It expects NPZ files containing `image` with shape `(112, 112, 10, 50)` and a ten-element `slice_valid_mask`, together with an authorized training manifest. For example, after preparing your manifest:

```bash
CUDA_VISIBLE_DEVICES=0 bash Align_Code/SaCMR_encoder/scripts/pretrain.sh \
  --device cuda:0 \
  --train_manifest SaCMR_encoder/manifests/train_npz.txt
```

This launcher forwards appended arguments and automatically enters `Align_Code`. Set the batch size, accumulation and worker count for your hardware. Further input validation and manifest conventions are documented in [the short-axis encoder README](Align_Code/SaCMR_encoder/README.md).

The short-axis teacher/alignment launchers are in `Align_Code/SaCMR_ECG_Align/scripts/`: `prepare_data.sh`, `prepare_henan_data.sh`, `train_cmr_sem_teacher.sh`, `stmem_sem_stage1.sh` and `stmem_sem_stage2.sh`. The preparation utilities consume supplied partitions; they do not generate the original participant assignment. They match paired acquisitions and check participant overlap and data integrity.

For initialization of the short-axis teacher, place `SaxCMR_PretrainedModel.pth` in `Align_Code/checkpoints/cmr/` and supply `--sa_pretrained_weights checkpoints/cmr/SaxCMR_PretrainedModel.pth` to that branch's teacher launcher. A downstream ECG configuration for `ECGPretrainedModel_with_SaxCMR.pth` should use `model_name: st_mem_vit_base`, `mode: align` and `afterAlign_path` pointing to the copied checkpoint, with the model dimensions expected by the downstream task.

**Upload preparation item:** the short-axis alignment modules import `Align_Code/util/sem_ablation.py`; this auxiliary file must be added from the working server before executing that branch. The supplied CMR MAE checkpoints do not include the subsequently trained semantic and phenotype heads.

## Data partitions and access

In the study, training, validation and test partitions were defined strictly at the participant EID level. All records belonging to the same participant were assigned to a single partition, with no participant overlap or participant-level leakage between the partitions.

UK Biobank participant EIDs cannot be distributed publicly under its data-access requirements, and the identifiers are specific to each approved application. The original UKB EID-bearing partition files are therefore not part of this public release. Approved users should prepare participant-disjoint tables using the identifiers supplied under their own UK Biobank application and the formats described above. The example MIMIC inference data do not constitute a release of the UKB partition files.

Clinical cohort data and other controlled-access datasets are not bundled with the source code. The presence of a relative data path is a placement convention, not an indication that the corresponding private dataset is included.

## Relative paths and other training weights

Run top-level `main_Cla*.py` scripts from the repository root. Their relative `data/`, `configs/`, `checkpoints/` and `outputs/` paths can be populated locally or overridden through the entry point's available arguments. Run direct alignment commands from `Align_Code`; use each baseline's README for its independent working-directory convention.

For non-SimpleTest downstream scripts, place the original ECGFM-KED checkpoint under `ECG_baselines/ECGFM_KED/`, FG-CLEP's `pytorch_model.bin` at `ECG_baselines/FG_CLEP/checkpoints/best/pytorch_model.bin`, and ST-MEM initialization weights at the location requested by the selected configuration. Some legacy downstream entries expect MERL's original weights as `ECG_baselines/MERL_ICML2024/vit_tiny_best_encoder.pth`; copy the released `MERL/best_encoder.pth` under that filename for those entries, while retaining `best_encoder.pth` for SimpleTest.

Top-level downstream CMR defaults use `checkpoints/cmr/pretrain_ep400_wep40_bs128_blr1e-3_mix_5x/checkpoint-399.pth`; either place `CMR_PretrainedModel.pth` there under the expected filename or pass its actual relative location through `--cmr_pretrained_weights`. Training configurations under `configs/Cla/` are experiment-specific and must be supplied or generated as appropriate; the two root YAML files belong to the SimpleTest example.

The documented inference commands explicitly set the configuration and output location. They also supply the FG-CLEP import path when needed. This allows the original validated SimpleTest program to be retained despite legacy defaults in its source.

## Citation

Please cite the manuscript and the specific [Hugging Face resource record](https://huggingface.co/Yukui1999/MyoECG) used for your work. Cite the original comparator papers when using those models and consult their linked repositories for bibliographic details.

Related work: Ding Z, Li Z, Hu Y, et al. Generating Cardiac Magnetic Resonance Images from Electrocardiograms—A Multicenter Study. *NEJM AI*. 2026;3(4).
