# XMatcher 推理接口与 Batch 支持

> 状态：**已支持 batch 推理**（`BaseMatcher.batch_call`）。LightGlue 和
> EfficientLoFTR 都实现了 `_forward_batch`。本文档解释入口、契约、各方法
> 怎么 batch 的、有哪些限制。

## 1. 入口层级

| 层级 | 入口 | 输入 | 输出 |
|---|---|---|---|
| CLI | `python -m xmatcher.cli.run` | YAML 配置 | `.npz` 文件 + `manifest.jsonl` |
| Python: 单对 | `build_matcher(cfg)(pair)` | `ImagePair` | `MatchResult` |
| Python: 多对 | `build_matcher(cfg).batch_call(pairs)` | `list[ImagePair]` | `list[MatchResult]` |
| 适配器内部: 单对 | `BaseMatcher._forward(pair)` | `ImagePair` | `_RawOutput` |
| 适配器内部: 多对 | `BaseMatcher._forward_batch(pairs)` | `list[ImagePair]` | `list[_RawOutput]` |

CLI 当前只走单对路径（一个 `for pair in dataset` 循环）。把 CLI 接到
`batch_call` 是后续工作（见 §6）。

## 2. 契约

### `BaseMatcher.__call__(pair)` (xmatcher/core/base.py:48)

```python
@torch.inference_mode()
def __call__(self, pair: ImagePair, *, return_processed_coords: bool = False) -> MatchResult:
```

单对入口，per-pair 默认实现。子类只需实现 `_forward`。

### `BaseMatcher.batch_call(pairs)` (xmatcher/core/base.py:62)

```python
@torch.inference_mode()
def batch_call(self, pairs: list[ImagePair], *, return_processed_coords: bool = False) -> list[MatchResult]:
```

**契约（合约性质，由 contract 测试锁定 `tests/contract/test_batch_call.py`）：**

- 输入空 list → 输出空 list。
- 输出顺序与 `pairs` 输入顺序一一对应（`results[i]` 对应 `pairs[i]`）。
- 若子类实现的 `_forward_batch` 返回数量不等于 `len(pairs)`，抛 `RuntimeError`。
- B=1 的 `batch_call([pair])` 行为应当与 `__call__(pair)` 数值一致（LightGlue 走 fast path 保证此性质；其他方法走 batch path 但模型对 B=1 行为通常稳定）。
- 后处理（`unproject` + `filter_by_mask`）按每对各自的 `meta` 应用——不会用第一对的 meta 处理所有对。
- `runtime_ms` 字段是 batch 总耗时除以 B，每对相同；这是已知的近似（不破坏每对单独计时的细节，但确实丢了细粒度信息）。

### `_forward(pair)` 与 `_forward_batch(pairs)` 的关系

- `_forward(pair)`：必填。
- `_forward_batch(pairs)`：可选。默认实现是循环 `_forward`：

```python
def _forward_batch(self, pairs):
    return [self._forward(p) for p in pairs]
```

子类有真正 batched 模型路径时覆写 `_forward_batch`，立刻获得 GPU 利用率红利。

## 3. 各方法的 batch 实现

### EfficientLoFTR (`xmatcher/methods/efficient_loftr.py`)

**完美 batch-friendly。** transformers 的 `AutoImageProcessor` 接收
`images=[[im0,im1], ...]` 嵌套 list，外层即 batch 维。`post_process_keypoint_matching`
镜像同样的结构。所以 batch 实现就是：

```python
def _forward_batch(self, pairs):
    images = [[p.image0, p.image1] for p in pairs]
    target_sizes = [[p.meta0.processed_size, p.meta1.processed_size] for p in pairs]
    inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
    outputs = self.model(**inputs)
    results = self.processor.post_process_keypoint_matching(
        outputs, target_sizes=target_sizes, threshold=self.params.match_threshold,
    )
    return [_RawOutput(...) for r in results]
```

`_forward` 现在就是 `_forward_batch([pair])[0]` 的薄包装。

**没有任何限制项**——同 batch 内图片尺寸不同也能跑（processor 内部 resize）。

### LightGlue (`xmatcher/methods/lightglue.py`)

**Batch 可行，但需要绕开上游一个 bug + 关一项自适应。**

**关键步骤**（参考 `vauto4d_lightning/.../end2end_gluer.py`）：

1. SuperPoint 提点：循环每张图 (2B 次) `extractor.extract(img)`。**不能直接** `extract(stack_of_images)`：
   - 上游 `extract` 第 141 行 `assert img.shape[0] == 1`；
   - 即使去掉 assert，`SuperPoint.forward` 内部对每张图独立 top-K + `torch.stack`，要求**所有图都恰好检出 K 个点**才能 stack 成 batch tensor。
2. **手动 pad 到统一 K**：`K = min(max_count_in_batch, max_num_keypoints)`，不足补零。
3. Stack 成 `(2B, K, 2)` / `(2B, K, D)`，按 `[::2] / [1::2]` 拆成 image0/image1 两个 batch。
4. **关掉 width pruning**：`self.matcher.conf.width_confidence = -1`（运行时临时设，finally 块恢复）。
5. 调用 matcher，输出 `out["matches"]` 已经是 `list[Tensor(Mi, 2)]`、`out["scores"]` 是 `list[Tensor(Mi)]`，每对一个。
6. Per-pair 索引出 `mkpts0/mkpts1`。

**B=1 时**直接走 `_forward(pair)` 单对路径，避免触发上面任何特殊处理。

#### 为什么关掉 width pruning？

LightGlue 的 width_confidence > 0 启用"point pruning" —— 在每一层动态丢弃 confidence 低的关键点。上游实现（`lightglue.py:538-541`）：

```python
ind0 = torch.arange(0, m, device=device)[None]   # shape (1, M) ← 写死了 batch=1
ind1 = torch.arange(0, n, device=device)[None]
prune0 = torch.ones_like(ind0)
prune1 = torch.ones_like(ind1)
```

后面用 `ind0[k, ...]` 索引 (k 在 0..B-1)，B>1 时 IndexError。这是一个**上游 bug**（GitHub issue 上有讨论但未修）。两条路：

1. **关 width pruning（我们的做法）**：单行 `width_confidence=-1`，零侵入。代价是丢失 width pruning 的 ~10-30% 加速。**depth pruning（早停）仍然生效**，那才是 LightGlue 大头的优化。
2. **patch 上游**（vauto4d 走的路）：把 `[None]` 改成 `[None].expand(b, -1).contiguous()` + 把 `index_select` 改成 per-batch 处理。需要维护一份 fork。

我们选 1。性能数据（L4，sample 图）：

| 配置 | 时间 | 匹配数 |
|---|---|---|
| per-pair (width prune on) | 38 ms | 277 |
| batch B=2 (width prune off) | 44 ms / 对 | 301 |

Batch 比 per-pair 略慢（关掉 width pruning 的代价大于 batch 收益），但匹配数略多（保留更多关键点）。

**结论**：LightGlue **不一定**从 batch 受益。`AutoModelForKeypointMatching` 的 transformers 包装版（`stevenbucaille/lightglue-superpoint`）是 batch-friendly 的，未来切到 transformers 路径时可去掉这一限制。

#### 同 batch 内尺寸限制

LightGlue 的 batch 路径要求**同 batch 内所有图 H/W 相同**。原因：SuperPoint 自己虽然按图独立提点（前一节循环），但 LightGlue 内部归一化坐标用的是单个 `image_size`；不同尺寸混 batch 会让坐标系错乱。

调用方需要在喂入 `batch_call` 前按 H/W group 好。MVP 的 `FromPairListDataset` 不做这件事；CLI 加 `--batch-size` 时需要配 sort-by-size 或 group-by-size。

EfficientLoFTR **没有此限制**——processor 自动 resize 到 480×640。

## 4. 性能数据（参考）

L4 GPU，`tests/fixtures/sample_a.jpg` (1024×731, RGB) + `sample_b.jpg` (768×1024, RGB)：

| method | mode | runtime | matches |
|---|---|---|---|
| LightGlue | per-pair | 38 ms | 277 |
| LightGlue | batch B=2 | 44 ms/对 | 301 |
| EfficientLoFTR | per-pair | 163 ms | 401 |
| EfficientLoFTR | batch B=2 | 163 ms/对 | 401 |

更大 batch、更多对的实战数据等真实数据集再做。

## 5. 风险点 & 注意事项

| 风险 | 影响 | 缓解 |
|---|---|---|
| LightGlue 同 batch 内图尺寸需统一 | 调用方要分组 | 加 `--batch-size` 时配 group-by-size |
| LightGlue 关闭 width pruning 后丢一点速度 | 个位数 ms | depth pruning 仍生效，影响有限 |
| GPU 显存随 B 线性增长 | OOM 风险 | 调用方控制 batch size，未来加 dynamic batching |
| `valid_mask` 后处理仍是 per-pair | 每对独立 filter | 已经在 `BaseMatcher._postprocess` 内部循环处理 |
| `runtime_ms` 是 batch 平均值 | 单对计时不准 | 调用方需要细粒度计时时用 `__call__`，不用 `batch_call` |

## 6. 后续工作

1. **CLI 接 `--batch-size`**：CLI run.py 加循环外的 chunk → `batch_call` 调用 → 写盘。
2. **DataLoader 多 worker prefetch**：当前 dataset 是 `Iterator[ImagePair]`，单线程读图。改成 `torch.utils.data.Dataset` + `DataLoader(num_workers=4)`，IO 与 GPU 并行。这一步对所有方法都生效，比 batch 收益更稳定。
3. **同尺寸 group**：dataset 配置加 `group_by_size: bool`，emit 时按 (H, W) 分桶。
4. **LightGlue 切到 transformers**：`stevenbucaille/lightglue-superpoint` 是上游 LightGlue 的官方 transformers 包装，自带正确的 batch 处理。切换后能拿掉 `width_confidence=-1` 这个 hack。

## 7. 关键文件索引

- `xmatcher/core/base.py:62` — `BaseMatcher.batch_call`
- `xmatcher/core/base.py:46` — `_forward_batch` 默认 fallback
- `xmatcher/core/base.py:101` — `_postprocess` 模板（`__call__` 与 `batch_call` 共用）
- `xmatcher/methods/lightglue.py:55` — LightGlue `_forward_batch`
- `xmatcher/methods/efficient_loftr.py:55` — EfficientLoFTR `_forward_batch`
- `tests/contract/test_batch_call.py` — batch 契约测试 (7 cases)
- `vauto4d_lightning/.../end2end_gluer.py` — 参考实现 (LightGlue batch)
