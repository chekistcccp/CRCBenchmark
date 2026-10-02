# 2026 年 CCF-A benchmark 对照与 ColoGround-Bench 发表准备

检索日期：2026-10-02。本文将文献中的具体设计与本项目的研究判断分开；缺口和优先级是本项目的评估，不是会议统一的录用条件。代码准备完成不代表这些研究证据已经取得。

## 检索范围及已确认的六篇正式论文

采用 [CCF 官网人工智能目录](https://www.ccf.org.cn/Academic_Evaluation/AI/) 中的 A 类会议 CVPR、ACL、AAAI；正式长文身份以会议论文库为准。未把 workshop、Findings、仅有 arXiv 预印本身份的工作计入六篇。部分论文在 2025 年已有预印本，其正式会议发表年份是 2026。

CVPR 论文通过 CVF 正式条目确认发表身份，并补读作者 arXiv 全文及其附录；ACL 和 AAAI 直接阅读正式论文 PDF。作者预印本与最终会议版可能有细节差异，因此不将两个版本不同的实验数值混用。

| 2026 正式论文 | 可核验的研究设计 | 对当前工作的启发 |
|---|---|---|
| [Med-CMR — CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Med-CMR_A_Fine-Grained_Benchmark_Integrating_Visual_Evidence_and_Clinical_Logic_CVPR_2026_paper.html) | 20,653 道题，分解视觉与推理能力；专家参与质量筛选；18 个模型；错误分析。全文：[作者版本](https://arxiv.org/html/2512.00818v1)。 | 需要解释每个任务测量什么，以及与已有医学 VQA benchmark 的差异；病例分割真值存在不等于新渲染题目已经得到临床核验。 |
| [OmniBrainBench — CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_OmniBrainBench_A_Comprehensive_Multimodal_Benchmark_for_Brain_Imaging_Analysis_Across_CVPR_2026_paper.html) | 9,527 道题、15 种成像模态；24 个模型；专业放射科核验与医生比较。全文：[作者版本](https://arxiv.org/html/2511.00846v1)。 | 不能仅凭模型低分断定任务有效；人工比较应基于与模型相同的输入信息。单病种也可以有价值，但需要说明其特有问题。 |
| [GroundingME — CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Li_GroundingME_Exposing_the_Visual_Grounding_Gap_in_MLLMs_through_Multi-Dimensional_CVPR_2026_paper.html) | 1,005 题、25 个模型；区分相似目标、空间关系、受限可见性和不可定位查询；人工核验。全文：[作者版本](https://arxiv.org/html/2512.17495v1)。 | 规模本身不是唯一贡献；有解释力的困难类型和拒答能力更重要。我们应同时报告阳性题、阴性题、格式失败及简单策略。 |
| [ScaleBench — ACL Long](https://aclanthology.org/2026.acl-long.1887/) | 6,574 题；明确标注指南与多轮检查；独立于标注者的读者在抽样子集上建立人工参照；开闭源模型比较。[正式 PDF](https://aclanthology.org/2026.acl-long.1887.pdf)。 | 公开图像可用于 benchmark，但需要题目质量、来源与任务独特性证据。抽样人工实验可以先检查可行性，不应称为整个数据集的人工上限。 |
| [ProgressLM / Progress-Bench — ACL Long](https://aclanthology.org/2026.acl-long.516/) | 3,325 个观察样本；同视角/跨视角、可回答/不可回答条件；14 个 VLM；另有域外人类活动实验。[正式 PDF](https://aclanthology.org/2026.acl-long.516.pdf)。 | 对照实验需要明确改变了什么变量；开发集的关联检验不能替代新队列上的泛化实验。训练改进属于该论文额外内容，不是所有 benchmark 必须照做的要求。 |
| [CMedBench — AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39264) | 五个评价维度同时涉及医疗能力、可信度与计算效率；研究压缩带来的权衡。[正式 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/39264/43225)。 | 应保存资源成本、版本与运行设置，使结果可复现。其压缩任务不同，不能直接套用为本项目必须开展压缩实验的理由。 |

## 对现有工作的判断

当前方向仍是 benchmark：通过真实分割标注构造结直肠 CT 检索、点框定位、连续切片判断和正常/肿瘤区域辨别，评价模型的能力与失败模式。已经具备患者隔离、固定 manifest、原始回答、严格解析、患者层面区间、非视觉基线和开发集提示敏感性分析。

不过，当前更接近“可复现的 benchmark 原型及失败诊断”，尚缺足够证据支撑一篇完成度较高的医学 benchmark 投稿。下一步应补齐测量有效性与研究定位；持续提高现有分数并不能解决这些问题。本文不预测会议录用概率，也不把文献中的模型数、题量或医生数当作统一门槛。

推荐研究表述：**基于分割标注的结直肠 CT 视觉证据 benchmark，在切片检索、定位与局部连续性任务中区分格式失败、位置先验和图像信息的作用。** T3 输入是连续九层的二维 montage，不能据此宣称评测了完整 3D 体积推理；T4 是标注引导的 ROI 比较，不能宣称端到端临床诊断。

## 按优先级排列的研究缺口

| 优先级 | 缺口 | 现有证据与不足 | 建议补充 |
|---|---|---|---|
| P0 | 题目可判读性与标注语义 | 分割真值有来源；CARE 数值语义来自用户确认；目前没有医生参与，无法确认缩放、裁剪、边界层的临床可见性。 | 对相同模型输入做独立临床盲评，记录不可判读及歧义原因；核验 CARE normal 标签的组织范围，尤其 T4 提示里的直肠区域描述。已备工具，真实评价仍未开展。 |
| P0 | 新颖性及与现有 benchmark 的差异 | 五轨道组合本身不足以证明贡献；医学 VQA 与 grounding benchmark 已大量存在。 | 增加现有数据/任务对照表，明确结直肠 CT、真实切片顺序、像素定位真值及先验对照带来的独特测量价值。 |
| P0 | 图像信息的作用 | 已有 36 组患者块置换检验；全部未显著。它检验已有回答与真值的关联，未实际改变模型输入。 | 开发集上比较原图、仅文字与中性图；固定题目和生成设置，配对比较分数与格式率。此次代码已实现，GPU 实验尚未运行。 |
| P1 | 失败来源解释 | Qwen3.5 大量格式失败；其他模型部分任务也失败；96 token 预算可能影响结果。 | 先分析图像对照的格式覆盖。如果全部条件都无法产生有效输出，只能报告协议完成失败；另行预先固定全模型预算敏感性实验，不能为每个模型试出最优提示。 |
| P1 | 强参照及模型范围 | 六个受 H20/H100 条件约束的开放 VLM；只有一个医疗模型家族；已有简单策略，没有临床人工参照。 | 先取得人工参照；如讨论“医疗专用模型整体是否更好”，补多个独立医疗家族。如讨论视觉定位能力，增加专用视觉方法作为单独标注训练设置的参照。闭源模型有助于定位前沿差距，但不是必选项。 |
| P1 | 难度与亚组解释 | 当前有病灶大小、边界分布和轨道覆盖信息，但缺完整系统性亚组报告。 | 预先固定像素病灶面积占比等非临床亚组，报告患者数与区间；未知 CARE spacing 不换算成毫米，不创造分期等临床标签。避免按模型分数挑“困难病例”。 |
| P1 | 可复现与发布 | manifest 指纹可靠；尚未完整归档精确权重版本、环境、硬件和资源开销；交接文档曾过时。 | 新补充记录输入图像/掩码字节哈希、运行包版本和模型配置；单独归档实际权重 revision/checksum、许可证及公开数据重建方法。元数据文件哈希不等于权重哈希。 |
| 条件性 | 独立泛化验证 | v2.4/v2.5 与后续诊断复用病例，属于事后修订和分析。 | 若论文声称独立泛化，必须新增未参与设计的队列并事先锁定方案。新医院队列不是所有 benchmark 投稿的通用硬门槛；公开病例也不能被称为已排除预训练接触。 |

目前没有医生是最主要的现实限制。可以继续完成自动对照、技术质控与结果报告；不能把模型裁决当成临床专家核验，也不能在论文中填入尚未开展的人工基线。

## 现有结果如何支持研究问题

- 正式 MSD T1 多个有效模型 Recall@3 约 0.258–0.277，接近随机基线 0.25；CARE Qwen3.6 达到 0.4829，但只覆盖 26 位 T1 合格评价患者，不能写成 61 位患者上的结果。应检验原图相对仅文字是否产生增量。
- T2 分数很低，同时有格式失败和定位失败。比如 Qwen3.6 的 Pointing Accuracy 为 MSD 0.0818、CARE 0.0929；需要确认医生在这些缩放图像上是否能找到目标，以及与中心/均匀点基线的差值。
- T3 全选九层具有高 F1；CARE Qwen3.6 的 0.7972 相对全选 0.7909 的配对区间包含零。CARE 开发位置先验 BA 约 0.5947，高于 GLM 的 0.5830 与 Qwen3.6 的 0.5389。不能把高 F1 自动解释为视觉定位。
- CARE T4 中 GLM、Qwen3.6 的准确率均约 0.5246；MedGemma 的回答全部无效。应结合两个交换方向、格式率和选边偏好解释，而不是把这些情况统一称为医疗视觉能力差。
- T5 原图 PRESENT 覆盖太低，当前不宜作为论文的核心因果或 faithfulness 贡献。其探索性数据仍可完整报告。

## 本次代码改动和验证边界

新增 `run_publication_supplement.sh`：默认准备开发集题目与人工包；显式 `EVIDENCE_CONTROL=1` 才进行 GPU 推理。补充在单独目录中进行。

`run_evidence_control.py` 为每个开发 T1–T4 原题生成三条件：原图、完全不传图片的 text_only、固定灰色背景并重绘 A–L/A–I/A–B 标签的 neutral_image。问题、真值、解码配置与 96 token 上限不变。原图也在同一补充运行中推理，保证同次运行条件下配对。中性图不复制源图像像素。输入和 T2 掩码字节哈希参与 manifest 指纹；改变输入、模型配置或运行包版本时拒绝复用已有回答。

`PipelineVLM.generate(None, prompt)` 构造仅文字输入。测试确认未传入假图片路径。若某模型不支持仅文字模式，错误使该实验保持未完成并保留已完成回答；不会将适配器异常计为错误答案或默默缩小模型组。实际六个模型的仅文字路径尚待服务器验证。

配对报告先按患者聚合，再计算原图减去控制条件的 bootstrap 区间，并展示全部条件、格式率及 T1 阴性特异度。区间是探索性结果，不据其挑选提示或声称经过多重比较校正的显著性。原图与控制的差值同时包含输入分布、格式遵循等影响；仅凭该差值不能证明模型定位了病灶。

人工工具生成 78 题（默认每数据集/轨道最多 10 位患者；CARE T1 只有 8 位合格开发患者；T4 保留两次交换）。读者只拿 `reviewer/`，浏览器可离线答题和导出；真值留在 `private/`。评分报告患者均值、格式率、题目可判读性及可判读性评级的一致率/kappa。小样本开发盲评是可行性实验，不能称为全数据集人工上限；不会据其改变正式主分数或自动删题。

已完成 47 项测试，真实 MSD/CARE 准备分别成功生成 522/486 道题；盲评页面生成 78 题，通过 JavaScript 语法检查，页面不含私有评分字段。用明确标记的模拟回答验证了人工评分链路。模拟回答仅用于软件测试，不是临床研究结果。新 GPU 推理、真实医生评分和精确权重归档仍未完成。

## 下一步可直接运行的实验

当前已有主实验、提示敏感性和患者置换分析，下一轮运行的是新增图像可用性对照。

```bash
cd /你的路径/CRCBenchmark
git pull --ff-only origin main
conda activate crcbench

# 不需要 GPU：先确认所有源图像/掩码齐全，准备题目与盲评包。
bash run_publication_supplement.sh

# 在已分配的一张 H20/H100 GPU 上：六模型顺序运行新增对照。
EVIDENCE_CONTROL=1 bash run_publication_supplement.sh
```

输出在 `runs/protocol_v2_5/supplement/publication/`：

```text
human_review/reviewer/review.html
human_review/private/review_key.json
evidence/<model>/<msd|care>/evidence_manifest.jsonl
evidence/<model>/<msd|care>/evidence_metadata.json
evidence/<model>/<msd|care>/snapshot_metadata.json
evidence/<model>/<msd|care>/hardware.json
evidence/<model>/<msd|care>/evidence_predictions.jsonl
evidence/<model>/<msd|care>/evidence_summary.json
```

本轮共 336 个原题 × 3 条件 × 6 模型 = **6,048 条预测**。MSD 每模型 522 条，CARE 每模型 486 条。不要用最终 JSONL 行数代替独立患者数；MSD 为 20 位开发患者，CARE 为 20 位，其中 T1 仅覆盖 8 位。

同命令可断点续跑。先准备后推理应在同一服务器环境及输出目录运行。如果迁移 artifacts 后 manifest 路径失效，可显式设置 `ARTIFACT_ROOT=/实际路径/runs/protocol_v2_5/artifacts`；该目录是 artifacts 本身，不是其父目录。迁移或换环境后使用新的 `OUTPUT_ROOT`，避免混用旧预测。

当前没有医生，可以只执行上述自动实验。未来有读者参与时，把 `human_review/reviewer/` 的独立副本交给每位读者，使用不同编号，拿回导出的 JSONL 后执行：

```bash
python scripts/evaluate_human_review.py \
  --key runs/protocol_v2_5/supplement/publication/human_review/private/review_key.json \
  --readers /路径/reader_1.jsonl /路径/reader_2.jsonl \
  --evidence-root runs/protocol_v2_5/supplement/publication/evidence \
  --output runs/protocol_v2_5/supplement/publication/human_review/human_summary.json
```

单读者可先验证可行性，但不能产生读者间一致性证据。模型与人工比较只使用该盲评子集，不能拿全评价集模型均值与开发人工子集直接比较。

## 下一批结果的分析决策

1. 核对全部条件的预测完整性与模型输入支持，区分适配器错误、格式失败和可评分的错误回答。
2. 查看原图相对两个控制的患者配对差值，并一起检查格式率。原图优势仅出现在格式率时，不解释为肿瘤识别改善。
3. 原图与两个控制相近时，报告该设置下先验/格式主导的证据范围；不继续据开发结果选提示，也不把未显著写成完全忽略图像。
4. 原图有增量时，仍需要标注语义和可判读性核验，再决定是否值得开展新队列验证、预先固定的预算/分辨率敏感性或专用视觉参照。
5. 完成数据卡、许可与重建步骤、版本和资源归档、按病例而非题目数量解释统计单位。P0 证据缺口未解决前，不能将工程完成状态描述为医学 benchmark 已达到发表要求。
