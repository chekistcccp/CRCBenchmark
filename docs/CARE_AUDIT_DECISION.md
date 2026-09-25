# CARE 数据审查决策记录

本文件记录 ColoGround-Bench 对当前公开 CARE release 的审查结论与主分析队列冻结决策。

## 1. 审查版本

已完成：

- audit v1：确认 CARE 不是原始 NIfTI/DICOM，而是二维 NPZ image-label pair；
- audit v2：确认文件名可恢复 case_id + source slice index，并发现 bbox CSV 与全部 NPZ 数量不一致；
- audit v3：系统比较 txt、bbox_txt、bbox_csv、all NPZ 四种 index source。

## 2. v3 核心结果

| 来源 | Train slices | Test slices | Train cases | Test cases | 唯一 case |
|---|---:|---:|---:|---:|---:|
| all NPZ | 26,656 | 6,461 | 318 | 81 | 399 |
| txt | 26,656 | 6,461 | 318 | 81 | 399 |
| bbox_txt | 26,537 | 6,424 | 318 | 81 | 399 |
| bbox_csv | 26,537 | 6,424 | 318 | 81 | 399 |

额外确认：

- train/test patient ID overlap = 0；
- 所有候选文件名都可以恢复 case_id + source slice index；
- test.txt 覆盖全部 6,461 个 test NPZ；
- train.txt 覆盖全部 26,656 个 train NPZ；
- bbox train 子集少 119 slices；
- bbox test 子集少 37 slices；
- 公开 release 的完整 train+test 组合无法同时复现论文报告的 33,024 slice pairs 与 398 patients。

## 3. Primary CARE cohort

主分析固定为：

```text
split        = test
index source = test.txt
patients     = 81
slices       = 6,461
```

理由：

1. test.txt 的 6,461 slices 与论文 test cohort 完全一致；
2. 81 patients 与论文 test cohort 完全一致；
3. 每个 list entry 都有对应 NPZ；
4. 所有 test NPZ 均被 test.txt 覆盖；
5. 不需要对不一致的 train release 做人工删减或猜测；
6. 所有 81 个 test patient 均存在至少一个连续 9-slice 局部窗口，可支持 T3。

## 4. 非主分析数据

### CARE train

仅用于：

- development；
- smoke test；
- debugging；
- supplementary sensitivity analysis。

不得与 test 合并后宣称“398/399 patient primary cohort”。

### bbox-defined subset

```text
train_bbox.csv / train_bbox.txt
test_bbox.csv  / test_bbox.txt
```

仅用于 supplementary sensitivity analysis。

## 5. 图像与标签处理

- CARE 图像按 release 中预处理后的 [0,1] 图像使用；
- 不假定其像素值为原始 HU；
- raw labels 可出现 0/1/2/3；
- 按官方 U-SAM dataloader 执行：
  `label[label > 2] = 2`
- canonical class 1/2 的医学语义在未被官方证据明确确认前仍不写死。

## 6. 当前默认命令

```bash
python scripts/index_datasets.py \
  --care-root data/extracted/CARE \
  --care-splits test \
  --care-index-source txt \
  --care-tumor-label <VERIFIED_ID> \
  --care-normal-label <VERIFIED_ID> \
  --output manifests/cases.jsonl
```

主 Benchmark 默认不得自动加入 CARE train。


## 7. 未决 class 语义的自动双分支实验

当前 release 能够确认 canonical foreground classes 为 1 和 2，但数值 ID 与“tumor / normal rectal tissue”的医学语义尚未获得足够明确的官方数字映射说明。

为避免这一点阻塞完整 benchmark，正式代码预先固定两套 CARE semantic sensitivity branches：

```text
care_tumor1_normal2
  tumor  = 1
  normal = 2

care_tumor2_normal1
  tumor  = 2
  normal = 1
```

两套分支：

- 使用同一 test.txt primary cohort；
- 分别生成独立 cases manifest；
- 分别生成独立 benchmark manifest；
- 使用独立 artifacts 目录；
- 所有模型分别运行；
- 所有结果分别评价。

两套结果不得用于“通过哪个模型表现更高来推测真实标签语义”。未来如获得权威映射，匹配的分支升级为 primary CARE result，反向分支作为 label-inversion sensitivity/control。

根目录 `run.sh` 已自动执行以上两套 CARE 分支，因此正式计算实验不再依赖手工设置 `CARE_TUMOR_LABEL` / `CARE_NORMAL_LABEL`。
