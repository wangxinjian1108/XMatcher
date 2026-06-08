# XMatcher: Mask 支持设计与状态

## 目标

允许调用方为每张图提供一个 `valid_mask`（前景/ROI/语义分割掩码），保证关键点
匹配只发生在 mask 标记为 True 的区域内。

## 三阶段流水线

按从输入到输出的顺序：

```
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 1: 输入预处理 mask                                     │
       │   把 valid_mask 之外的像素置零                               │
       │   作用点: BaseMatcher.__call__ / batch_call (template 层)    │
       │   状态: ✅ 已实现 (apply_input_mask)                         │
       └──────────────────────────────────────────────────────────────┘
                                     ↓
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 2: 检测后过滤 (post-detection filter)                  │
       │   关键点检测之后，丢弃落在 mask 外的关键点                   │
       │   作用点: LightGlue 适配器 _forward / _forward_batch         │
       │   状态:                                                      │
       │     - LightGlue ✅ 已实现 (_filter_feats_by_mask)            │
       │     - EfficientLoFTR N/A (detector-free, 没这个阶段)         │
       └──────────────────────────────────────────────────────────────┘
                                     ↓
       ┌──────────────────────────────────────────────────────────────┐
       │ Stage 3: Matcher attention mask  ← TODO                      │
       │   在 self/cross-attention 内部把 mask 外位置的注意力关掉     │
       │   作用点: matcher 内部 transformer                           │
       │   状态: ⏸ 暂不实现, see "Stage 3 TODO" below                 │
       └──────────────────────────────────────────────────────────────┘
                                     ↓
       ┌──────────────────────────────────────────────────────────────┐
       │ 兜底: post-match filter                                      │
       │   最终匹配点再过一遍 mask, 再 unproject 到原图坐标系         │
       │   作用点: BaseMatcher._postprocess (template 层)             │
       │   状态: ✅ 已实现 (filter_by_mask, MVP 一开始就有)           │
       └──────────────────────────────────────────────────────────────┘
```

不同方法走不同阶段：

| 方法 | Stage 1 | Stage 2 | Stage 3 | 兜底 |
|---|---|---|---|---|
| LightGlue (有 detector) | ✅ | ✅ 已实现 | ⏸ 不需要 (Stage 2 已过滤) | ✅ |
| EfficientLoFTR (detector-free) | ✅ | N/A | ⏸ TODO | ✅ |

## 调用方契约

把 segmentation mask（前景 = True）和 padding mask（有效像素 = True）做 AND，
写入 `PreprocessMeta.valid_mask`：

```python
from xmatcher.core.types import PreprocessMeta, ImagePair

fg_mask = ...        # (H, W) bool, your segmentation
padding_ok = ...     # (H, W) bool, True where preprocessing didn't pad
valid_mask = fg_mask & padding_ok    # 单一字段管两件事

meta = PreprocessMeta(
    original_size=(H_orig, W_orig),
    processed_size=(H_proc, W_proc),
    crop_box=None, scale=(1.0, 1.0), pad=(0, 0, 0, 0),
    valid_mask=valid_mask,
)
pair = ImagePair(image0=img0, image1=img1, meta0=meta, meta1=meta_other, ...)
match_result = matcher(pair)
# match_result.mkpts0 / mkpts1 全部位于各自 mask 内部
```

`valid_mask` shape 必须等于 `processed_size`。`PreprocessMeta.__post_init__`
会强制校验。

## 已实现的部分

### Stage 1: `apply_input_mask`

`xmatcher/core/preprocess.py`:

```python
def apply_input_mask(image: torch.Tensor, meta: PreprocessMeta) -> torch.Tensor:
    """Zero out pixels of `image` that fall outside `meta.valid_mask`.
    Returns a new tensor; original is untouched. No-op when mask is None."""
```

`BaseMatcher.__call__` 和 `BaseMatcher.batch_call` 在调用 `_forward` /
`_forward_batch` 之前都会先把 `pair.image0/image1` 通过 `apply_input_mask`
处理一遍，然后传给适配器。原始 `pair` 不被修改。

### Stage 2: `_filter_feats_by_mask` (LightGlue 内部)

`xmatcher/methods/lightglue.py`:

```python
def _filter_feats_by_mask(feats: dict, meta) -> dict:
    """Drop keypoints/descriptors/scores whose (u, v) lands outside meta.valid_mask."""
```

LightGlue 的 `_forward` 在 `extractor.extract(...)` 之后立即调用此函数：

- 单对：filter feats0/feats1 → 喂给 matcher。
- 批量：每张图 extract 后立即 filter，再去 pad 到统一 K 准备 batch。

关键设计：**Stage 1 已经把 mask 外像素置零，所以 SuperPoint 在那里几乎不会
检出强响应。但 mask 边界附近仍可能产生 spurious detections。Stage 2 把
"keypoints 必须严格在 mask 内"作为硬契约**。

### 兜底: `filter_by_mask`

`BaseMatcher._postprocess` 用 `filter_by_mask` 再过一遍最终匹配点。这一层
对 detector-free 的 EfficientLoFTR 尤其重要——它没 Stage 2，前两个阶段
（输入 mask + post-match filter）就是它能依赖的全部。

## Stage 3 TODO: matcher 内部 attention mask

### 为什么暂不实现

1. **LightGlue 不需要 Stage 3**——Stage 2 已经把 mask 外的关键点完全丢掉，
   matcher 只看到 mask 内的点。继续在 matcher 里 mask 是冗余的。
2. **EfficientLoFTR 需要但难做**——它是 detector-free，密集网格匹配。
   要在 attention 阶段 mask 必须改 transformers `EfficientLoFTRModel`
   内部 forward pass，给它注入 attention_mask。这违反我们"不 patch 上游"
   的原则。
3. **现状已有兜底**——EfficientLoFTR 走"输入 mask + post-match filter"
   双层保护，对绝大多数应用够用。Stage 3 只在两种场景下有意义：
   - mask 内/外特征语义混淆，post-filter 之后得到的匹配数太少；
   - 想避免 attention 阶段把 mask 外区域的 token 拉进计算（性能优化）。

### 真要做的话

可选路径：

**方案 A: Patch transformers 上游（侵入式）**

派生 `EfficientLoFTRForKeypointMatching`，重写 `forward` 接受
`attention_mask` 参数，把 mask 下采样到 coarse / fine 各阶段的网格分辨率，
传给每一层 attention。需要维护一份 fork 跟着 transformers 4.51-4.54 走。

**方案 B: 输入图像区域涂掉（目前的轻量做法）**

`apply_input_mask` 已经做了——把 mask 外像素归零。模型的 backbone 看到
零图就提不出强 feature，attention 自然不会聚焦在那里。代价是边界附近
feature 被污染（zero pixel 不是 "no info" 而是黑色）。

**方案 C: 等待上游支持**

HuggingFace 可能给 KeypointMatching pipeline 加原生 mask 支持，但目前
没看到 RFC。

### 决策记录

- 当前选 **方案 B**（输入 mask + post-match filter，零 patch 上游）。
- 如果用户报告"mask 边界附近的伪匹配仍然存在"，再上方案 A。
- Patch 进 thirdparty 之前先开 issue 评估收益（多少匹配质量提升 vs 维护
  fork 的成本）。

### 验证 stage 3 是否真的需要

下面是一组实验，评估 Stage 3 是否值得做：

1. 用一组带前景 mask 的 dataset 跑 EfficientLoFTR：
   - 现状（Stage 1 + post-match filter）的匹配数和质量
   - 假设你 fork 上游加了 attention mask 后的匹配数和质量
2. 如果两者匹配数 / 几何精度差异 < 5%，**不要实现 Stage 3**。
3. 如果差异显著，按方案 A patch。

数据集需要：
- 真实场景下的前景 mask（如 SAM 输出）
- 同场景两个视角的 image pair
- pose ground truth（计算 reprojection error 评估匹配质量）

## 测试

`tests/unit/test_mask_helpers.py` 覆盖 `apply_input_mask` +
`filter_kpts_by_mask` 的所有边界情况：no-op、mask shape、设备、dtype、
索引顺序等。10 个 case。

L4 实测（`tests/fixtures/sample_a.jpg` + `sample_b.jpg`，top-half mask）：

| method | 全图匹配数 | mask 内匹配数 | mask 外匹配数 |
|---|---|---|---|
| LightGlue per-pair | 247 | 58 | 0 |
| LightGlue batch B=2 | 247 | 65 | 0 |
| EfficientLoFTR per-pair | 186 | 78 | 0 |
| EfficientLoFTR batch B=2 | 8 (\*) | 2 (\*) | 0 |

(\*) EfficientLoFTR batch 模式本身就有质量损失（与 mask 无关），见
`INFERENCE_INTERFACE_AND_BATCHING.md` Section 5。

**所有方法、所有模式下，mask 外匹配数 = 0**——硬契约成立。

## 关键文件索引

- `xmatcher/core/types.py` — `PreprocessMeta.valid_mask` 字段
- `xmatcher/core/preprocess.py` — `apply_input_mask`, `filter_kpts_by_mask`,
  `filter_by_mask` (post-match)
- `xmatcher/core/base.py:_apply_pair_input_mask` — Stage 1 wiring
- `xmatcher/methods/lightglue.py:_filter_feats_by_mask` — Stage 2 (LightGlue)
- `tests/unit/test_mask_helpers.py` — Stage 1/2 单元测试
