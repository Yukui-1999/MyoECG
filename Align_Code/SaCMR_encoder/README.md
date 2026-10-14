# 短轴多切片 CMR：切片先融合的 MAE

本目录实现方案 A，保持 50 个心动周期帧作为输入通道。

```text
NPZ image                     [112,112,10,50]
只缩放 XY，整理 batch          [B,10,50,96,96]
各有效层共享 patch projection  [B,10,36,768]
先移除各层相同的空间 patch     [B,10,9,768]     (mask_ratio=0.75)
带切片位置的注意力池化         [B,9,768]
加 CLS，ViT-B 编码             [B,10,768]
共享的切片条件解码器           重建有效切片中被遮挡的空间 patch
```

提取表征时不遮挡，主干处理 36 个空间 token 加 CLS，`forward_features` 返回 `[B,768]`。编码器为 768 维、12 层、12 头；解码器为 256 维、2 层、8 头。切片位置编码与切片注意力均可训练。整个体积的融合特征参与逐层重建，因此融合模块在 MAE 预训练中有梯度。

## 启动预训练

在仓库根目录运行：

```bash
cd Align_Code
bash SaCMR_encoder/scripts/pretrain.sh
```

本目录的相对路径以 `Align_Code/` 为基准。启动脚本会自动定位该目录；直接调用 Python 时，请先进入 `Align_Code/`。脚本默认使用当前环境中的 `python`，可通过 `SACMR_PYTHON` 指定其他解释器。数据和预训练权重需要自行放入对应的 `data/`、`checkpoints/` 目录；这些文件不随代码提供。

脚本使用 `python`，GPU 3，当前患者 batch 为 256、梯度累积 1 次（保留用户修改），有效 batch 为 256，训练 400 epochs、warmup 40 epochs、遮挡率 75%。默认 base LR 为 `1e-3`，实际 LR 按有效 batch / 256 缩放，当前为 `1e-3`。AdamW 的 betas 为 `(0.9,0.95)`。默认开启 CUDA FP16 AMP、梯度检查点和分组解码；该入口目前支持单进程、单 GPU。

启动脚本默认使用 `--num_workers 8 --prefetch_factor 1`，在加载并行度与主机内存占用之间折中；直接调用 Python 时 `num_workers` 默认仍为 4。CPU 图像和空间插值仍为 float32，原有传入 GPU 时的 FP16 转换及 AMP 流程保持不变。一个 256 人的 float32 batch 约占 4.39 GiB；原来的 16 worker、每 worker 预取 2 个 batch 可容纳约 140.6 GiB 图像，脚本当前设置为约 35.2 GiB。这是预取图像容量，不是整个训练进程的内存上限：组装 batch、当前 batch、固定内存副本/缓存和其他进程还会占用内存。启动时的 `configuration.data_loader` 会打印该估算。

worker 数和预取量只影响加载并行度，不改变样本顺序、batch、损失或学习率；可以在断点续训时调整。主机仍有内存压力时可继续使用 `--num_workers 2 --prefetch_factor 1`。`--num_workers 0` 会关闭多进程预取。

`num_workers` 增加并行解压、检查和缩放数据的进程数；`prefetch_factor` 增加每个 worker 提前准备的 batch 数，用于缓冲加载时间的波动。若 worker 持续忙而 GPU 在等待，应先增加 worker；仅增加预取量不能提高持续的数据处理能力。当前 8/1 仍慢且内存余量充足时可用 `--num_workers 12 --prefetch_factor 1`（约 52.7 GiB 预取图像容量）。8/2 与 16/1 的预取图像容量均约 70.3 GiB，但实际总内存仍会不同，不能只按该乘积预测总占用。共享服务器无法仅靠这两个参数保证整机占用低于 350 GiB；应持续观察 `MemAvailable`，在约 503 GiB 总内存的主机上，保留约 153 GiB available 对应约 350 GiB 的“总内存减可用内存”。

末尾传参可以覆盖脚本默认值，例如：

```bash
bash SaCMR_encoder/scripts/pretrain.sh \
  --device cuda:1 \
  --batch_size 8 --accum_iter 8 \
  --output_dir SaCMR_encoder/outputs/sa_mae_run2
```

也可以直接调用 `python -m SaCMR_encoder.main_pretrain` 或 `python SaCMR_encoder/main_pretrain.py`；直接调用时须指定 `--data_root`。使用 `--help` 查看参数。

启动脚本立即输出启动提示，随后报告环境检查、数据索引、模型初始化和就绪状态；启动期间每 10 秒输出一次 heartbeat。迁移到 NVMe 后，默认通过显式清单固定原来的 54,388 个样本，逐项检查路径是否存在，不从仍在复制中的目录生成新队列。进入训练时输出 `epoch_start`，第一个 batch 完成后立即输出损失。请等数据复制命令成功结束后再启动。

自动发现数据时只解析一次数据根目录，已规范化的样本路径直接复用于队列指纹和清单，避免逐文件重复访问 NFS 元数据。首次索引仍需枚举受试者目录。正在运行的旧进程不会自动加载这些修改；重新启动后生效。

## 数据与训练队列

默认数据根目录是 `Align_Code/data/processed_sa_x112_y10`。脚本通过 `$SACMR_PROJECT_ROOT/data/processed_sa_x112_y10` 定位数据，并使用 `SaCMR_encoder/manifests/train_manifest_nvme.txt`。请自行提供数据及清单；可在清单中使用相对数据根目录的路径。程序运行时生成的规范化绝对路径仅用于本机索引、队列核对和断点续训，不是代码中固定的机器路径。训练输入清单独立于 `outputs/` 保存，清理训练输出不会删除它。每个文件必须包含：

- `image`：`(112,112,10,50)`；沿 XY 双线性缩放到 96，保持层序与帧序。
- `slice_valid_mask`：长度 10，1 表示有效，0 表示无效。0 可以出现在任意位置。

数据按文件中已保存的强度直接使用，当前数据为归一化到 `[0,1]` 的 float16。加载与空间插值在 CPU 上使用 float32，CUDA AMP 训练时以 float16 传入 GPU。无效切片先置零，模型还会把它们排除在 patch 编码、注意力和重建损失之外；不会依靠无效切片的像素值判断有效性。形状错误、非二值掩码、全无效样本或有效区域的非有限数值会报错。

直接调用 Python 入口且不指定 `--train_manifest` 或 `--auto_train_manifest` 时，使用目录内全部扫描到的样本，包括同一受试者的不同访视。若要沿用已有训练/验证/测试划分，请传入仅包含预训练队列的清单，同一受试者的不同访视应放在同一集合：

```text
<EID_A>/<EID_A>_20209_2_0_sa_processed.npz
<EID_B>/<EID_B>_20209_2_0_sa_processed.npz
```

上述 EID_A、EID_B、EID_C 均为占位符；使用者需替换为自己获批数据中的标识符，不表示公开原始 UKB EID。每行一个相对数据根目录的路径或绝对路径，不带表头：

```bash
bash SaCMR_encoder/scripts/pretrain.sh --train_manifest SaCMR_encoder/manifests/train_npz.txt
```

显式的 `--train_manifest` 优先于自动清单，清单缺失时会报错。若直接调用 Python 入口并指定 `--auto_train_manifest SaCMR_encoder/manifests/new_manifest.txt --auto_manifest_validation full`，则支持生成并复用完整检查清单；此时不要同时传入 `--train_manifest`。生成时采用数据目录当时的内容，将摘要及排除原因写入同名 `.audit.json`。完整模式要求已有清单具有匹配的完整检查报告。

启动后在输出目录保存实际使用的 `train_manifest.txt` 和路径列表指纹，恢复时核对队列，避免目录新增文件导致续训数据发生静默变化。清单仅锁定路径，不校验文件内容。

本程序生成的清单使用已规范化的绝对路径，可以添加 `--manifest_is_canonical` 直接复用，跳过逐文件的 NFS 路径解析和存在性预检查：

```bash
bash SaCMR_encoder/scripts/pretrain.sh \
  --train_manifest SaCMR_encoder/manifests/train_manifest.txt \
  --manifest_is_canonical \
  --output_dir SaCMR_encoder/outputs/another_run
```

该选项只适用于确认已经规范化的绝对路径清单；相对路径、包含 `..` 等条目会被拒绝。文件仍会在读取时验证，因此它不代表忽略空文件或损坏数据。

0 字节 `.npz` 文件没有图像内容。NumPy 原始错误可能显示 `Cannot load file containing pickled data when allow_pickle=False`，这不表示应开启 pickle。读取器现在会直接报告文件完整路径及 `empty file (0 bytes)`，并为损坏 ZIP/压缩数据保留原始错误原因。需要重新生成这些数据，或在单独记录排除项后使用过滤清单。

2026-09-13 文件大小检查发现 54,458 个文件中有 69 个空文件，过滤后剩余 54,389 个路径。但非空文件也可能被截断：`<EID_C>/<EID_C>_20209_3_0_sa_processed.npz` 在本地与原 NAS 上均为 4 MiB，带 ZIP 文件头但无法读取 ZIP 目录。旧的 `train_manifest_nonempty_npz_local.txt` 只代表大小检查，保留用于对照。

完整检查工具会实际解压 `image` 和 `slice_valid_mask`（读取必需成员时验证 ZIP CRC），并复用训练读取器检查形状、二值且至少一层有效的掩码、有效切片的有限数值。未使用的附加字段不解压，不开启 pickle。原数据的坏样本路径及原因记录在 `manifests/train_manifest_validated_npz_local.audit.json`；文件访问错误或内存不足会终止检查，不会作为坏数据静默排除。原始 NPZ 文件不会被修改。

原数据全量检查结果：54,458 个文件中，69 个为空文件，1 个为上述非空损坏 ZIP；54,388 个样本通过完整读取验证，检查用时约 38 分钟。NVMe 迁移清单仅替换路径前缀，来源及指纹记录在 `manifests/train_manifest_nvme.relocation.json`，此记录不表示已确认复制完整或重新验证了目标文件内容。原清单与原审计报告保留。

复制完成后，默认启动命令即可使用迁移清单，也可以显式指定：

```bash
bash SaCMR_encoder/scripts/pretrain.sh \
  --train_manifest SaCMR_encoder/manifests/train_manifest_nvme.txt \
  --output_dir SaCMR_encoder/outputs/mae_vit_base_patch16_validated
```

复制完成后，如需对目标数据重新完整检查，可单独运行（这会实际解压所有样本）：

```bash
python -u -m SaCMR_encoder.data.manifest \
  --data_root data/processed_sa_x112_y10 \
  --manifest SaCMR_encoder/manifests/train_manifest_nvme_rechecked.txt \
  --workers 4
```

检查完成后可用 `--train_manifest SaCMR_encoder/manifests/train_manifest_nvme_rechecked.txt` 指定复检结果。清单复用不监控文件后续变化；若数据被替换或重新生成，应指定新的清单路径重新检查。

当前已有训练检查点。迁移后继续已有训练时，使用下文的 `--resume` 命令；从头训练则指定新的 `--output_dir`。

## 损失与显存

`slice_valid_mask=1` 表示有效切片；模型返回的 `reconstruction_mask=1` 表示被遮挡的空间 patch，两者含义不同。所有切片共享同一张空间遮挡图，遮挡内容在切片融合前移除。先对每位患者的被遮挡 patch 和有效切片求平均，再对患者求平均，避免 10 层患者比 8 层患者权重更大。

默认 `return_predictions=False`，仅预测隐藏 patch 的像素。按最多 2 个切片位置一组计算解码与损失，并对整组计算使用梯度检查点；这一做法避免仅写循环却仍保留全部解码激活。`--decoder_slice_chunk_size 1` 可进一步缩小解码组，`--no_checkpoint` 用更多显存减少重算。输入数据与 patch projection 开销仍随有效层数增加。

完整预测主要用于检查或可视化，会增加显存：

```python
import torch
from SaCMR_encoder.models_mae import mae_vit_base_patch16

model = mae_vit_base_patch16()
# images: [B,10,50,96,96], valid: [B,10]
loss, predictions, spatial_mask = model(images, valid, return_predictions=True)
# predictions: [B,10,36,12800]; invalid slices are zero-filled.
reconstruction = model.unpatchify(predictions)  # [B,10,50,96,96]

model.eval()
with torch.no_grad():
    features = model.forward_features(images, valid)  # [B,768]
```

## 输出与断点续训

默认输出到 `SaCMR_encoder/outputs/mae_vit_base_patch16_validated/`：

- `config.json`：配置；`train_manifest.txt`：实际样本路径。
- `log.jsonl`：每个 epoch 的损失、样本数、学习率、优化器更新尝试次数、耗时和 CUDA 张量峰值显存等；`optimizer_steps` 包含 AMP 溢出时跳过的更新尝试。
- `checkpoint-latest.pth`：每个 epoch 保存的最新状态。
- `checkpoint-XXXX.pth`：默认每 20 epochs 以及最后一个 epoch 保存，文件编号为从 0 开始的 epoch 索引。

续训保持原命令的训练参数，并添加：

```bash
bash SaCMR_encoder/scripts/pretrain.sh \
  --resume SaCMR_encoder/outputs/mae_vit_base_patch16_validated/checkpoint-latest.pth
```

数据根目录迁移时，续训会将当前清单的相对样本路径映射回检查点中的旧根目录，并核对完整有序队列指纹；只有样本及顺序完全一致才接受路径变化。此检查不访问旧盘，不更改原检查点，也不跳过 batch、学习率等其他配置核对。迁移续训使用脚本中的新清单；旧输出目录里的清单快照可能仍是旧盘绝对路径，不要用它覆盖新清单。复制完成后，请先结束旧训练进程，再运行续训命令。

若最初覆盖了 batch、学习率、seed、输出目录等，恢复时也使用相同值。可改变 GPU、worker 数、预取量、梯度检查点开关、解码分组大小。恢复包含模型、优化器、AMP scaler 和随机状态，从已保存 epoch 的下一轮开始。改变总 epochs 会明确提示余弦学习率周期随之改变。中途未完成 epoch 的进度不保存。新的训练应使用新的输出目录。

从现有长轴 MAE 初始化（可选，与 `--resume` 互斥）：

```bash
bash SaCMR_encoder/scripts/pretrain.sh --init_checkpoint checkpoints/cmr/lax_mae_checkpoint.pth
```

只加载形状匹配的 patch projection、空间位置、CLS、ViT blocks 和最终归一化；新的切片融合与轻量解码器从头训练。模型会报告加载和跳过的权重。

## 验证

```bash
python -m unittest discover -s SaCMR_encoder/tests -v
```

快速检查完整入口（在独立输出目录执行，只有少量样本/批次）：

```bash
bash SaCMR_encoder/scripts/pretrain.sh \
  --model mae_vit_tiny_patch16 \
  --max_samples 2 --max_steps 1 \
  --batch_size 2 --accum_iter 1 \
  --epochs 1 --warmup_epochs 0 --num_workers 0 \
  --output_dir SaCMR_encoder/outputs/sa_mae_smoke_run
```

`mae_vit_tiny_patch16` 仅用于快速测试，正式训练默认使用 `mae_vit_base_patch16`。`--max_steps` 限制每个 epoch 的 batch 数；正式训练不传 `--max_samples/--max_steps`。验证结果与测量条件见 `docs/verification.md`。
