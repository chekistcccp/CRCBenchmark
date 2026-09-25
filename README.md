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

## 2. 当前 CARE 实际审查结论

CARE 已完成两轮只读审查。v2 对真实下载的 `CARE.zip` 得到以下事实：

- 数据根目录：`DataV6`；
- 全部 NPZ：train **26,656**，test **6,461**；
- 所有 NPZ 文件名均可 100% 解析为 `case_id + slice_index`；
- 按“全部 NPZ”统计：train **318** 个 case ID、test **81** 个 case ID；
- train/test case ID **没有重叠**，因此全部 NPZ 实际形成 **399 个唯一 case ID**，与论文所述总患者数 398 不一致；
- CARE 论文摘要写 317 train + 81 test，而 Methods 段又写 318 train + 81 test，因此正式 Benchmark 不用“全部 NPZ case 数”自行修正论文，而是继续核对 release 自带的正式 slice list；
- train 中 318/318 病例均至少存在连续 9 层窗口，test 中 81/81 亦如此，因此 CARE 在结构上可以支持局部 T3；
- 图像均为预处理后的 `512×512`、值域 `[0,1]`，不重新作为 HU 执行 CT window；
- raw label 抽样值为 `0/1/2/3`；按照官方 U-SAM dataloader，`label > 2` 会被并入 canonical class 2，因此本项目同样采用 `3 → 2`；
- canonical class 1/2 的**医学含义仍需独立确认**，目前代码不会默认把二者写死为 normal/tumor。

v2 还确认：`bbox.csv` **并不覆盖全部 NPZ**。

- train `train_bbox.csv`：26,537 条，全部都能找到对应 NPZ，但另有 **119 个 NPZ 不在 CSV 中**；
- test `test_bbox.csv`：6,424 条，全部都能找到对应 NPZ，但另有 **37 个 NPZ 不在 CSV 中**。

因此，现在不能再简单把 `bbox.csv` 当作 ColoGround-Bench 的唯一正式 slice list。压缩包同时提供：

```text
train/train.txt
train/train_bbox.txt
train/train_bbox.csv

test/test.txt
test/test_bbox.txt
test/test_bbox.csv
```

其中 `train.txt/test.txt` 很可能更接近论文所述的完整 image-label pair 列表，但必须先用 **audit v3** 精确比较四套索引来源：

1. `txt`
2. `bbox_txt`
3. `bbox_csv`
4. 全部 `npz`

新版审查会同时比较：

- train/test slice 数；
- patient 数；
- train/test patient overlap；
- 是否与论文 26,563 + 6,461 slice pairs 对齐；
- 是否与论文 398 patients 对齐；
- 每个列表是否引用缺失 NPZ；
- 不同列表相对于全部 NPZ 的增删集合。

运行：

```bash
git pull

CARE_SOURCE=data/raw/CARE/CARE.zip \
bash run_data_audit.sh
```

默认生成：

```text
manifests/care_audit_v3.json
```

### CARE 正式索引安全门

正式索引器现在支持：

```text
--care-index-source txt
--care-index-source bbox_csv
--care-index-source all_npz
```

默认：

```text
--care-index-source auto
```

但当 `train.txt` 与 `train_bbox.csv` 内容不同，`auto` 会**主动停止**，不会替研究者静默选择。只有 v3 审查完成后，才把 primary CARE index source 冻结到协议中。

当前 CARE 的科学状态可以概括为：

> **患者 ID / source slice index 已具有很强的可恢复证据，T1 与局部 T3 在结构上可行；但正式纳入哪些 slice，以及 canonical class 1/2 的医学语义，还需最后一次冻结。**

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

当前设计以**4 × RTX 3090 24 GB** 为主要运行环境。

小模型优先采用多 GPU 数据并行分片推理；不进行任务特异性训练或微调。

模型权重默认通过 **ModelScope** 下载到本地后运行。

---

## 5. 推荐环境

建议：

- Linux
- CUDA
- Python 3.11
- NVIDIA RTX 3090 × 4

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
  --care-mapping data/extracted/CARE/care_index.csv \
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

# 16. 下载模型

全部模型：

```bash
python scripts/download_models.py \
  --config configs/models.yaml \
  --model-root models
```

单独下载一个模型，例如 Qwen3.5-9B：

```bash
python scripts/download_models.py \
  --models qwen35_9b \
  --model-root models
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

# 18. 四张 RTX 3090 并行推理

对于可以单卡加载的模型，推荐按病例分片：

```bash
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g \
  python scripts/run_inference.py \
    --model qwen35_9b \
    --manifest manifests/benchmark_v1.jsonl \
    --shard-index $g \
    --num-shards 4 &
done

wait
```

然后合并：

```bash
python scripts/merge_shards.py \
  --dir predictions/qwen35_9b \
  --output predictions/qwen35_9b/all.jsonl
```

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

# 20. 一键运行

在数据审查完成并确认环境变量后：

```bash
export MSD_ROOT=/absolute/path/to/MSD

# 只有 CARE 完成审查后再启用：
# export CARE_ROOT=/absolute/path/to/CARE
# export CARE_MAPPING=/absolute/path/to/care_index.csv
# export CARE_TUMOR_LABEL=<VERIFIED_ID>
# export CARE_NORMAL_LABEL=<VERIFIED_ID>

bash run_benchmark.sh
```

如果设置了 `CARE_ROOT`，却没有显式设置：

```text
CARE_TUMOR_LABEL
```

程序会主动停止，而不是猜测。

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
- [x] 4-GPU 分片推理框架
- [x] 评价框架
- [x] CARE.zip v1/v2 审计
- [ ] CARE 标签医学语义最终确认
- [x] CARE filename 中 case_id / source slice index 可恢复
- [ ] CARE primary index source（txt / bbox_csv / all_npz）最终冻结
- [ ] Benchmark v1 manifest 冻结
- [ ] 全模型正式实验
