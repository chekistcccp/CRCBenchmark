# CRCBenchmark / ColoGround-Bench

**中文（默认）** | [English](README_EN.md)

ColoGround-Bench 是一个面向**结直肠癌 CT** 的开放权重多模态大模型（Vision-Language Model, VLM）**零样本 / training-free Benchmark**。本项目的核心目标不是测试模型“会不会回答医学问题”，而是系统评估模型是否真正：

- 找到了结直肠肿瘤；
- 将视觉证据定位在真实肿瘤区域；
- 理解病灶在连续 CT 层面上的出现与消失；
- 区分肿瘤与正常直肠壁；
- 在移除真实病灶后显著改变判断，而不是依赖数据集风格或其他 shortcut。

Benchmark 使用真实专家分割标注作为核心 ground truth，不人为构造 T 分期、病理、MSI、预后、坏死或其他数据集中不存在的临床标签。

---

## 0. 一键运行入口

正式实验统一使用：

```bash
bash run.sh
```

你不需要手动：

- 解压两个数据集；
- 建立病例索引；
- 手工指定 CARE class 1/2 的唯一语义；
- 下载模型；
- 逐个启动模型；
- 手工评价结果。

`run.sh` 会自动执行：

```text
检查 H20/H100 与 Python 环境
        ↓
自动发现 / 解压 MSD 与 CARE
        ↓
MSD Benchmark 生成一次
        ↓
CARE 语义分支 A：tumor=1, normal=2
        ↓
CARE 语义分支 B：tumor=2, normal=1
        ↓
自动下载 configs/models.yaml 中全部模型权重
        ↓
单卡顺序运行全部模型 × 全部实验分支
        ↓
自动评价
        ↓
汇总 results/all_experiments_summary.json
```

### Slurm 使用原则

本仓库**不提供也不调用任何 Slurm 提交脚本**：

```text
sbatch
srun
salloc
```

都由你自己管理。

你手动提交/进入已经分配好单张 H20 或 H100 的 Slurm 作业后，只需：

```bash
conda activate crcbench
cd CRCBenchmark
bash run.sh
```

`run.sh` 会使用作业环境已经设置好的 `CUDA_VISIBLE_DEVICES`。

---

## 0.1 数据集应该放哪里

最推荐的方式是**直接放压缩包，不要手动整理内部文件**：

```text
CRCBenchmark/
└── data/
    └── raw/
        ├── MSD/
        │   └── Task10_Colon.tar
        └── CARE/
            └── CARE.zip
```

MSD 文件名不强制必须叫 `Task10_Colon.tar`；只要放在：

```text
data/raw/MSD/
```

并且格式属于：

```text
.zip
.tar
.tar.gz
.tgz
```

即可。

CARE 推荐保持：

```text
data/raw/CARE/CARE.zip
```

执行：

```bash
bash run.sh
```

后，程序自动解压到：

```text
data/extracted/MSD/
data/extracted/CARE/
```

并自动寻找真正的数据根目录：

- MSD：包含 `imagesTr/` 和 `labelsTr/` 的目录；
- CARE：包含 `test/test_npz/` 和 `test/test.txt` 的目录。

如果 Slurm 作业因时限中断，重新运行 `bash run.sh` 时，已经完整解压的文件会按大小检查后跳过，不会从零重复解压整个 CARE。

CARE 解压后体积较大，建议服务器至少预留 **200–250 GB** 可用磁盘空间用于数据、模型、benchmark artifacts 和预测结果。

### 已经手动解压的数据也支持

如果你已经有：

```text
data/MSD/
├── imagesTr/
└── labelsTr/
```

程序会直接使用，不再解压 MSD。

也可以显式指定路径：

```bash
MSD_ROOT=/path/to/Task10_Colon \
CARE_ROOT=/path/to/DataV6 \
bash run.sh
```

---

## 0.2 CARE class 1 / class 2 不再阻塞实验

正式自动实验会同时构建两个 CARE 分支：

### 分支 A

```text
care_tumor1_normal2
tumor label  = 1
normal label = 2
```

### 分支 B

```text
care_tumor2_normal1
tumor label  = 2
normal label = 1
```

两套分支使用同一个冻结的 CARE primary cohort：

```text
test.txt
81 patients
6,461 slices
```

raw label 中的 `3` 仍按官方 U-SAM 规则：

```text
3 → 2
```

两套分支都会完整构建并运行相应的 T1–T5。

**重要：**

这两个分支是 semantic sensitivity analysis。不能根据“哪个分支模型成绩更高”来判断真正的 class 语义。

如果后续通过官方标注说明确认了正确映射：

- 正确分支作为 CARE primary result；
- 另一分支保留为 sensitivity / label-inversion control。

---

## 1. 项目定位

本项目使用两个公开数据集，但二者承担不同角色：

### MSD Task10 Colon

完整 3D NIfTI CT + 原发结肠癌分割掩膜。

主要用于：

- T1 病灶层面检索；
- T2 视觉定位；
- T3 三维连续性；
- T5 反事实视觉忠实性。

### CARE

公开 CARE 压缩包按照官方 U-SAM 的数据读取方式，当前首先按**预处理后的二维 NPZ 图像-标签对**进行审查。

在完成实际数据审计前，本项目**不会默认假设**：

- CARE 可以恢复完整 3D volume；
- NPZ 中的像素值是原始 HU；
- 标签值 1/2 分别对应哪一种医学组织；
- 连续数字文件名属于同一患者；
- 相邻文件一定对应相邻 CT 层面。

因此，CARE 必须执行“**先审查、后启用**”流程。

---

## 2. CARE v3 审查结论与正式 Benchmark 决策

CARE 已完成 v1–v3 三轮只读审查。v3 说明公开 release 中实际上存在两组不同用途的 slice list：

- `train.txt/test.txt`：等价于全部 NPZ 列表；
- `train_bbox.txt/test_bbox.txt`：与 bbox 子集数量一致；
- `train_bbox.csv/test_bbox.csv`：bbox 子集；
- 全部 NPZ。

实际数量为：

| 来源 | Train slices | Test slices | Train cases | Test cases | 唯一 case |
|---|---:|---:|---:|---:|---:|
| all NPZ | 26,656 | 6,461 | 318 | 81 | 399 |
| `txt` | 26,656 | 6,461 | 318 | 81 | 399 |
| `bbox_txt` | 26,537 | 6,424 | 318 | 81 | 399 |
| `bbox_csv` | 26,537 | 6,424 | 318 | 81 | 399 |

所有候选来源的 train/test case ID 都没有重叠，但**没有任何一套完整 train+test 索引能够同时复现论文报告的 33,024 slice pairs 和 398 patients**。因此 audit v3 不再把某一来源错误地标记成“唯一最佳来源”。

### Primary CARE cohort：冻结为官方 test split

为了避免把公开 release 的 train 版本差异带入主要结论，ColoGround-Bench 的 **primary CARE cohort 现在固定为：**

```text
split        = test
index source = test.txt
patients     = 81
slices       = 6,461
```

理由：

1. `test.txt` 有 **6,461** 条唯一记录；
2. 6,461 条全部都有对应 NPZ；
3. 它覆盖 test 目录中的全部 6,461 NPZ；
4. 文件名可 100% 恢复 `case_id + source slice index`；
5. 共得到 **81 个 test case**；
6. 6,461 slices / 81 cases 与论文公开的 test cohort 数量完全一致；
7. 所有 81 个 test case 都存在至少一个真实连续的 9-slice 局部窗口，可支持 T3。

因此 CARE train split **不进入 primary benchmark**。它仍保留为：

- 开发/调试集；
- supplementary sensitivity analysis；
- 后续解释 release 与论文数量差异的辅助数据。

### bbox 子集

`test_bbox.csv/test_bbox.txt` 只有 6,424 slices，比正式 test list 少 37 张；train bbox 子集比全 train list 少 119 张。因此 bbox 子集不作为 primary CARE cohort，但可以作为 supplementary sensitivity analysis：

```text
CARE_INDEX_SOURCE=bbox_csv
CARE_SPLITS=test
```

### 标签与图像处理

- 图像为 `512×512`、值域 `[0,1]` 的预处理图像，不重新当作 HU 做 CT window；
- raw labels 抽样观察到 `0/1/2/3`；
- 与官方 U-SAM 一致，代码执行 `label > 2 → 2`，得到 canonical `0/1/2`；
- canonical class 1/2 的医学语义仍需在正式 T1/T2/T4/T5 前完成最终确认，因此当前仍保留显式 label ID 安全门。

### 当前默认 CARE 索引行为

正式索引默认已经改为：

```text
--care-splits test
--care-index-source txt
```

只有显式要求时才会纳入 train：

```bash
python scripts/index_datasets.py \
  --care-root data/extracted/CARE \
  --care-splits train test \
  --care-index-source txt \
  --care-tumor-label <VERIFIED_ID> \
  --care-normal-label <VERIFIED_ID>
```

这意味着 CARE 的**数据结构审查阶段已经结束**。后续不再需要为了 318/317 或 399/398 的 release 差异不断修改 primary cohort；该差异会在论文 Methods / Limitations 中透明报告，而主要 CARE 实验固定使用可完全复现的 81-case test cohort。

---

## 2.1 你现在应该做什么

### 1. 更新代码

```bash
git pull
```

### 2. 建立一次完整环境

```bash
conda create -n crcbench python=3.11 -y
conda activate crcbench

pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

### 3. 把压缩包放到固定位置

```text
data/raw/MSD/<MSD压缩包>
data/raw/CARE/CARE.zip
```

### 4. 在你自己提交好的 Slurm 单卡作业中检查 GPU

```bash
nvidia-smi
python scripts/check_gpu.py
pytest -q
```

### 5. 一次性运行全部正式实验

```bash
bash run.sh
```

不需要再手动填写：

```text
CARE_TUMOR_LABEL
CARE_NORMAL_LABEL
```

因为两种映射都会分别运行。

---

## 3. Benchmark 五个 Track

| Track | 主要问题 | 数据集 | 主要指标 |
|---|---|---|---|
| **T1 Lesion Retrieval** | VLM 能否从同一病例的一组 CT 层面中找到肿瘤层面？ | MSD；CARE 在标签语义确认后启用 | Recall@3 |
| **T2 Visual Grounding** | VLM 能否把坐标或框真正定位到肿瘤？ | MSD + 经验证后的 CARE | Pointing Accuracy |
| **T3 Volumetric Consistency** | VLM 能否识别病灶沿 z 轴的出现/消失和连续性？ | MSD；CARE 仅使用 source slice index 真正连续的局部窗口 | Slice F1 |
| **T4 CARE Hard Negative** | VLM 能否区分癌性直肠组织与正常直肠壁？ | CARE | Pairwise Accuracy |
| **T5 Counterfactual Faithfulness** | 删除真实病灶是否比删除匹配对照区域更显著地改变模型判断？ | MSD + 经验证后的 CARE | Faithfulness Gap |

本项目**不设置人为加权的 Overall Score**。不同能力分别报告，避免用一个总分掩盖“分类正确但定位错误”等重要现象。

---

## 4. 计划评测的开放权重模型

当前 `configs/models.yaml` 包含：

| Key | 模型 |
|---|---|
| `qwen35_9b` | Qwen3.5-9B |
| `internvl3_8b` | InternVL3-8B |
| `minicpm_v_45` | MiniCPM-V-4.5 |
| `qwen25vl_7b` | Qwen2.5-VL-7B-Instruct |
| `medgemma_4b_it` | MedGemma-4B-IT |
| `lingshu_7b` | Lingshu-7B |

当前默认运行环境调整为**单张 NVIDIA H20 或 H100**。

- 默认只使用 1 张 GPU；
- 模型保持 BF16，不使用量化；
- 当前 4B–9B 级模型按单卡顺序运行；
- 多 GPU 分片仍保留为可选功能，但不再是默认实验条件；
- 不进行任务特异性训练或微调。

模型权重默认通过 **ModelScope** 下载到本地后运行。

---

## 5. 推荐环境

建议：

- Linux
- CUDA
- Python 3.11
- NVIDIA H20 × 1 或 H100 × 1
- BF16 推理

创建环境：

```bash
conda create -n crcbench python=3.11 -y
conda activate crcbench

pip install -r requirements.txt

export PYTHONPATH=$PWD/src:$PYTHONPATH
```

---

## 6. 推荐目录结构

```text
CRCBenchmark/
├── configs/
├── scripts/
├── src/
├── tests/
├── manifests/
├── artifacts/
├── predictions/
├── results/
├── models/
└── data/
    ├── MSD/
    ├── raw/
    │   └── CARE/
    │       └── CARE.zip
    └── extracted/
        └── CARE/
```

`data/`、模型权重、生成的 benchmark 图像和预测结果均已设计为不提交至 GitHub。

---

# 7. MSD Task10 Colon 数据准备

推荐结构：

```text
data/MSD/
├── imagesTr/
│   ├── colon_001.nii.gz
│   ├── colon_002.nii.gz
│   └── ...
└── labelsTr/
    ├── colon_001.nii.gz
    ├── colon_002.nii.gz
    └── ...
```

MSD 可以直接建立病例索引：

```bash
python scripts/index_datasets.py \
  --msd-root data/MSD \
  --output manifests/cases.jsonl
```

MSD 按原始 CT 处理，可以使用 HU soft-tissue window。

---

# 8. CARE：下载后第一步不是解压，而是审查

将下载得到的压缩包原封不动放入：

```text
data/raw/CARE/CARE.zip
```

然后直接执行：

```bash
CARE_SOURCE=data/raw/CARE/CARE.zip bash run_data_audit.sh
```

或：

```bash
python scripts/inspect_care.py \
  data/raw/CARE/CARE.zip \
  --sample-n 50 \
  --output manifests/care_audit_v3.json
```

该脚本是**只读审查工具**：

- 不会修改 CARE.zip；
- 不需要先完整解压几十 GB 数据；
- 不会自动把标签 1/2 赋予医学含义；
- 不会根据文件数字顺序自行构造 3D volume。

---

## 9. CARE 审查脚本会检查什么？

审查报告包括：

### 压缩包结构

- 总文件数量；
- train/test NPZ 数量；
- 是否存在：
  - `train_bbox.csv`
  - `test_bbox.csv`
  - `train_npz/`
  - `test_npz/`

### NPZ 内容

抽样读取 `.npz`：

- keys；
- `image.shape`；
- `image.dtype`；
- image min/max；
- `label.shape`；
- `label.dtype`；
- 实际出现的 label values。

例如可能观察到：

```text
label values: [0, 1, 2]
```

但程序会明确标记：

```text
SEMANTICS UNVERIFIED
```

因为仅看到 0/1/2 并不能证明：

```text
1 = normal wall
2 = tumor
```

或反过来。

---

## 10. CARE 是否可以恢复患者级 3D 序列？

这是正式 Benchmark 前必须解决的关键问题。

审查脚本会尝试以**保守规则**分析文件名能否证明：

```text
case_id + slice_index
```

例如下面这种形式可能具有恢复价值：

```text
patient001_0031.npz
patient001_0032.npz
patient001_0033.npz
```

而如果只是：

```text
1002.npz
1003.npz
1004.npz
```

程序**不会**假设它们属于同一患者，也不会认为它们是连续 CT 层。

审查结果会输出：

```text
patient/slice parse: ...
inferred cases: ...
multi-slice cases: ...
contiguous_multi_slice_fraction: ...
```

同时默认安全状态始终为：

```text
label_semantics_verified: false
patient_slice_mapping_verified: false
care_3d_tracks_enabled: false
```

这些 `false` 是科学安全机制，不是程序错误。

---

# 11. CARE 完整解压

第一轮 ZIP 审查完成后，再根据需要完整解压：

```bash
mkdir -p data/extracted/CARE

unzip \
  data/raw/CARE/CARE.zip \
  -d data/extracted/CARE
```

解压后再次检查：

```bash
python scripts/inspect_care.py \
  data/extracted/CARE \
  --sample-n 50 \
  --output manifests/care_audit_extracted.json
```

脚本会自动搜索额外的包装目录，因此类似：

```text
data/extracted/CARE/CARE/train/...
```

也可以识别。

---

# 12. CARE 患者/层序映射

如果文件名无法可靠恢复患者身份和切片顺序，则必须提供显式 mapping CSV：

```csv
case_id,slice_index,npz_path,split,slice_spacing,pixel_spacing_y,pixel_spacing_x
patient001,0,train/train_npz/xxx.npz,train,1.25,0.75,0.75
patient001,1,train/train_npz/yyy.npz,train,1.25,0.75,0.75
```

至少需要：

```text
case_id
slice_index
npz_path
split
```

如果没有经过验证的 patient/slice mapping：

- CARE 不参与 T1；
- CARE 不参与 T3；
- 不允许把每张 slice 当作独立患者进行论文统计；
- 仍可以在合理条件下用于 T2/T4/T5。

---

# 13. CARE 标签语义必须显式确认

当前代码已经取消所有 CARE 标签默认值。

只有完成官方标签语义确认以后，才能执行：

```bash
python scripts/index_datasets.py \
  --msd-root data/MSD \
  --care-root data/extracted/CARE \
  --care-splits test \
  --care-index-source txt \
  --care-tumor-label <已确认的标签ID> \
  --care-normal-label <已确认的标签ID> \
  --output manifests/cases.jsonl
```

如果 CARE 文件名已经经过人工验证，确实能够编码患者和层序，则可以省略：

```bash
--care-mapping
```

但仍必须显式提供 CARE 的医学标签 ID。

---

# 14. CARE 图像强度处理原则

### MSD

MSD 是原始 CT，因此使用 HU window，例如：

```text
WL = 50
WW = 400
```

### CARE

CARE 公共 NPZ 当前按**预处理后的打包图像**处理。

如果数组已位于：

```text
[0, 1]
```

则直接映射到 8-bit。

如果数值范围不是 0–1，但又没有证据表明它是原始 HU，则采用 robust display scaling。

**不会把 CARE 数组强行当作 HU 做 WL/WW。**

这是为了避免对已经预处理过的数据再次错误 windowing。

---

# 15. 构建 Benchmark

完成病例索引后：

```bash
python scripts/build_benchmark.py \
  --cases manifests/cases.jsonl \
  --config configs/benchmark.yaml \
  --output manifests/benchmark_v1.jsonl
```

生成的图像、mask 辅助文件和反事实图像放在：

```text
artifacts/
```

模型不会看到 GT segmentation overlay。

---

# 16. 模型权重自动下载

正式实验**不需要手动下载模型**。

执行：

```bash
bash run.sh
```

时会自动下载 `configs/models.yaml` 中的全部模型：

```text
qwen35_9b
internvl3_8b
minicpm_v_45
qwen25vl_7b
medgemma_4b_it
lingshu_7b
```

权重默认保存在：

```text
models/
```

下载器会自动复用已经完整下载的 snapshot，并对不完整的权重分片执行续传/重新获取。

仅在调试时才需要单独执行：

```bash
python scripts/download_models.py --models qwen35_9b
```

---

# 17. 单 GPU 推理

例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_inference.py \
  --model qwen35_9b \
  --manifest manifests/benchmark_v1.jsonl
```

---

# 18. 单卡 H20 / H100 推理

先确认服务器：

```bash
nvidia-smi
python scripts/check_gpu.py
```

默认使用单张 GPU：

```bash
export NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 \
python scripts/run_inference.py \
  --model qwen35_9b \
  --manifest manifests/benchmark_v1.jsonl
```

一键脚本也已经默认：

```text
NUM_GPUS=1
```

因此直接执行：

```bash
bash run.sh
```

不会再启动 4 个并行 worker。

正式实验直接执行：

```bash
bash run.sh
```

默认会运行 `configs/models.yaml` 中的**全部模型**，并在单张 H20/H100 上按顺序执行，因此不同模型不会同时占用显存。

只有在代码调试阶段才建议临时限制模型：

```bash
ONLY_MODELS="qwen35_9b" \
bash run.sh
```

`ONLY_MODELS` 不属于正式完整实验设置。

---

# 19. 结果评价

```bash
python scripts/evaluate.py \
  --manifest manifests/benchmark_v1.jsonl \
  --predictions predictions/qwen35_9b/all.jsonl \
  --output-dir results/qwen35_9b
```

核心评价遵循：

- patient-level statistics；
- 2,000 次 patient-level bootstrap；
- 不使用 LLM-as-a-Judge；
- 不人工修改模型输出；
- invalid output 保留为 failure；
- 不构造人为加权总分。

---

# 20. 一键运行全部实验

默认：

```bash
bash run.sh
```

会运行：

```text
MSD
CARE tumor=1 / normal=2
CARE tumor=2 / normal=1
```

并对 `configs/models.yaml` 中的全部模型完成推理与评价。

临时只调试 MSD：

```bash
ENABLE_CARE=0 bash run.sh
```

临时只调试一个模型：

```bash
ONLY_MODELS="qwen35_9b" bash run.sh
```

但正式完整实验应保持 `ONLY_MODELS` 为空。

---

# 21. 当前最重要的操作顺序

如果你刚开始准备数据，建议严格按照下面顺序：

```text
下载 MSD + CARE
       │
       ├── MSD → 解压 → 检查 imagesTr / labelsTr
       │
       └── CARE.zip 保持原样
                    │
                    ▼
            run_data_audit.sh
                    │
                    ▼
          manifests/care_audit_v3.json
                    │
                    ▼
      确认 CARE label 实际含义
                    │
                    ▼
   确认 patient ID / slice order 是否可恢复
                    │
                    ▼
       必要时建立 care_index.csv
                    │
                    ▼
         再启用 CARE Benchmark
                    │
                    ▼
           冻结 benchmark manifest
                    │
                    ▼
             运行全部 VLM
```

**当前阶段不要直接运行全量 CARE Benchmark。**

先完成：

```bash
CARE_SOURCE=data/raw/CARE/CARE.zip bash run_data_audit.sh
```

然后重点检查：

```text
manifests/care_audit_v3.json
```

---

# 22. 科学设计边界

本项目明确不做：

- T stage 推断；
- N stage 推断；
- MSI/KRAS/BRAF 推断；
- 病理等级推断；
- 生存/复发预测；
- 人工构造“高侵袭性”标签；
- 从 texture 人工定义“坏死”等病理结局；
- 使用分割 mask 提示 VLM 答案；
- 使用长 Chain-of-Thought 作为核心指标；
- 使用另一个 LLM 给模型解释打分。

Benchmark 的核心是：

```text
Did the model answer correctly?
                ↓
Did it look at the correct lesion?
                ↓
Did removing that lesion actually change its decision?
```

即：

**Recognition → Grounding → Faithfulness**

---

# 23. 论文方法学原则

为了保证后续能够用于高水平期刊投稿：

1. 所有模型运行同一个冻结后的 benchmark manifest；
2. ground-truth segmentation 只用于样本构建和评分；
3. patient 是主要统计学独立单位；
4. CARE 的患者关系不通过文件数字顺序猜测；
5. CARE 的标签医学含义不通过数值 ID 猜测；
6. CARE 与 MSD 因数据表示不同，应优先分别报告结果，而不是简单混成一个“跨中心总分”；
7. 动态反事实图像在运行 benchmark 时生成，用于测试真正的 lesion-specific evidence dependence；
8. benchmark 本身是研究主体，具体某一个 Qwen/InternVL 模型只是被测对象。

---

# 24. 详细研究协议

完整科学方案见：

[PROTOCOL.md](PROTOCOL.md)

CARE 下载后首先阅读：

```text
scripts/inspect_care.py
run_data_audit.sh
```

---

## 当前状态

- [x] Benchmark 五个 Track 框架
- [x] MSD NIfTI 数据索引
- [x] CARE NPZ 数据接口
- [x] CARE ZIP 只读审查工具
- [x] CARE 标签语义安全门
- [x] CARE patient/slice mapping 安全门
- [x] Benchmark 构建框架
- [x] 多模型配置
- [x] 单卡 H20/H100 顺序推理框架
- [x] 评价框架
- [x] 根目录 run.sh 全自动实验入口
- [x] 全模型权重自动下载与断点检查
- [x] 数据压缩包自动发现与可恢复解压
- [x] MSD + 两套 CARE 语义分支自动运行
- [x] CARE.zip v1/v2/v3 审计
- [x] CARE 两种 class 1/2 语义分支自动实验
- [x] CARE filename 中 case_id / source slice index 可恢复
- [x] CARE primary cohort 冻结：test split + test.txt（81 cases / 6,461 slices）
- [ ] Benchmark v1 manifest 冻结
- [ ] 全模型正式实验
