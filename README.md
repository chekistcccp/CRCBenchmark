# ColoGround-Bench / CRCBenchmark

面向结直肠 CT 的零样本视觉语言模型 benchmark。五个轨道分别检验肿瘤层面检索（T1）、点与框定位（T2）、连续层面识别（T3）、肿瘤与正常组织配对识别（T4）和探索性病灶扰动反应（T5）。任务使用真实分割标注，不构造分期、病理或预后标签。

完整实验规范见 [PROTOCOL.md](PROTOCOL.md)。旧版运行说明保存在 [v2.3 归档](docs/README_V2_3_ARCHIVE.md)，其 CARE 双分支结果不会被新版覆盖。

## 协议 v2.5

- **MSD Task10 Colon**：NIfTI CT 与肿瘤掩码，支持 T1/T2/T3/T5。
- **CARE 发布版 test 集**：81 名患者、6,461 张切片。按用户确认的标签语义，`0=背景`、`1=正常组织`、所有其他前景值为肿瘤；读取时把大于 1 的值统一映射为 canonical 类别 2。CARE 支持符合构题条件的 T1–T5。
- 两套数据均采用固定 SHA-256 患者划分：MSD 20 名开发患者、106 名正式评价患者；CARE 20 名开发患者、61 名正式评价患者。`PILOT_ONLY=1` 只在开发患者上检查输出格式。
- v2.3 曾用于适配器预检的 MSD `colon_001` 和 CARE `case17105001` 固定留在开发集，避免进入正式评价。
- 正式评价只读取 `benchmark_split=eval` 的题目，并要求预测题目完整、唯一且 manifest 指纹一致。
- T3 的真实边界在连续九层中的 C–G 位置按固定种子选择，消除 v2.4 固定 E 的位置捷径。构题审计在推理前核对患者隔离、任务覆盖和边界位置分布。
- T1/T2/T3 同时报告随机选层、均匀/中心点、全选九层/固定中心层等简单基线。T4 报告准确率、有效配对覆盖、两次都正确比例、交换一致性和选边偏好。所有主要分数都计入无效回答的失败；仅在格式有效回答上计算的分数单独作为次要指标。T5 首先报告原图 PRESENT 识别率；在没有经验证的候选似然分数前，只报告决策式扰动反应。
- v2.5 默认输出到 `runs/protocol_v2_5/`，保留 v2.3 与 v2.4 原始结果。v2.4 的 T3 与 v2.5 使用不同题目，不直接比较分数。

CARE 映射来自本项目用户的明确确认；项目尚未从发布方文档独立取得 NPZ 整数标签对照表。论文中应如实标注这一来源。

## 在已分配 GPU 的作业中运行

仓库不提交 Slurm 作业，也不调用 `sbatch`、`srun` 或 `salloc`。用户在一张 H20/H100 GPU 的作业环境中执行：

```bash
cd CRCBenchmark
conda activate crcbench
PILOT_ONLY=1 bash run.sh   # 可选：仅检查开发患者上的格式
bash run.sh                # 正式实验，自动继续已有的完整预测
```

原始压缩包可放于 `data/raw/MSD/` 和 `data/raw/CARE/CARE.zip`；也可设置 `MSD_ROOT`、`CARE_ROOT` 指向已解压目录。`run.sh` 自动准备数据、构建题目、下载六个模型、逐模型推理和评价。`ONLY_MODELS="..."` 仅供调试，不用于正式模型组比较。

默认结果：

```text
runs/protocol_v2_5/
├── manifests/      # 病例索引、患者划分、开发/评价题目、覆盖与构题审计
├── artifacts/      # 输入图像和标注掩码；开发与评价目录分离
├── predictions/    # 原始模型回答和 manifest 指纹
└── results/        # 逐题、逐患者及 bootstrap 汇总
```

开发集预检会在 `results/pilot/` 留下逐轨道格式报告。预检通过只表示适配器可生成至少一个合格格式的回答；正式结果仍分别统计无效率和视觉指标。所有模型采用贪心生成、同一冻结评价题目；不从普通解释文字猜测答案。

## 环境

使用 Python 3.11、PyTorch 2.13.0、torchvision 0.28.0、CUDA 12.6。先从 PyTorch cu126 wheel 安装 torch/torchvision，再运行 `pip install -r requirements.txt`；`requirements.txt` 故意不包含 torch。`run.sh` 的 preflight 会验证 GPU 与运行时，不会自行更换 PyTorch。

## 解释结果

benchmark 的任务是可靠测量能力和失败模式，不要求模型超过基线。报告时请分别列出数据覆盖、格式合规率、患者层面置信区间与简单策略基线。MSD 使用公开带标注训练集，无法排除模型预训练阶段的数据接触；CARE 是预处理后的 NPZ，而非已证实的原始 HU。T5 的连续 Faithfulness Gap 仍待候选似然读取方法验证。
