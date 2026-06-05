# XMatcher · 统一关键点匹配接口设计

| 字段 | 内容 |
|------|------|
| 状态 | Draft（待评审） |
| 创建日期 | 2026-06-06 |
| 范围 | MVP：LightGlue + EfficientLoFTR；后续扩展 RoMa v2 / XFeat |
| 关联文档 | `docs/design/UNIFIED_INTERFACE_KPTS_MATCHING.md`（原始备忘） |

## 1. 目标与范围

### 1.1 目标

提供一个统一的、配置驱动的关键点匹配运行接口，使下游能够：

- 通过 YAML 配置切换不同关键点匹配方法（MVP：LightGlue / EfficientLoFTR）。
- 输入"图像对 + 预处理元信息"，输出"原图坐标系下的稀疏对应点"。
- 把每种方法的依赖、模型私货、坐标变换都封装在适配器内部，对调用方透明。
- 提供一个统一的 Docker 镜像与构建管线，使本地开发与 CI/远程运行环境一致。

### 1.2 范围（MVP）

- 接入两个方法：`lightglue`、`efficient_loftr`。
- 提供 CLI（`xmatcher.cli.run`）+ Python API（`build_matcher`）。
- 提供 Docker 镜像 + GitHub Actions 自动构建推送至 GHCR。
- 提供权重下载脚本 + 锁文件。
- 提供 unit / contract / smoke 三层测试。

### 1.3 非目标（明确排除）

- 不实现完整 Dataset（库内仅提供 `FromPairListDataset` 这一个最小参考实现，正式 dataset 由外部仓库实现）。
- 不实现匹配可视化子命令（`xmatcher visualize` 留给后续阶段）。
- 不实现 GT 评测指标（mconf 不归一化，AUC/EPE 等指标在外部）。
- 不接 RoMa v2 / XFeat（架构预留，MVP 不实现）。
- 不解决离线场景下 LightGlue 自动下载失败的问题（首次跑要联网）。
- CI 不跑 smoke 测试（不上 self-hosted GPU runner）。

## 2. 架构总览

### 2.1 关键决策一览

| 决策 | 选择 | 理由 |
|------|------|------|
| 输出形态 | 统一稀疏 + 可选 dense 透传 | 稀疏对所有方法公约数；dense 留给 RoMa 等不丢能力 |
| 模块层次 | 单层 Matcher，内部自拼 detector | 避免给 detector-free 方法加 NullDetector 假皮 |
| Dataset 范围 | 只交付预处理后图像对 + meta；外部实现真 dataset | 库聚焦 matcher，不绑死数据组织形式 |
| 预处理位置 | Dataset 直接交付预处理后图，meta 描述变换链 | 调用链明确；后处理只剩"反变换" |
| 后处理默认 | 默认应用 unproject 与 mask 过滤；输出原图坐标 | 下游一致体验 |
| 集成方式 | Submodule + sys.path 注入 + Python 适配器 | 保持上游原貌；不动上游可减少维护成本 |
| 入口 | CLI + YAML 配置（method 配置与 dataset 配置分两份） | 复现性好；dataset 复用率高 |
| Docker | 单一镜像装齐所有方法；代码烤进 `/app`；推 GHCR | 一致性 + 可发布 |
| 权重 | 脚本下载到 `~/.cache/xmatcher`；缺失硬错误 | 显式优于隐式；镜像不带权重 |
| 测试 | unit + contract（无 GPU/权重）+ smoke（本地 GPU）| CI 友好；契约可锁定 |
| 配置校验 | pydantic（外层 + 各方法 ParamModel） | 错误提示精准 |
| 输出格式 | 每对 `.npz` + 全局 `manifest.jsonl` | 跨语言、调试友好、不绑 PyTorch |

### 2.2 总体目录

```
XMatcher/
  xmatcher/
    __init__.py
    core/
      types.py                  # ImagePair, MatchResult, DenseField, PreprocessMeta
      base.py                   # BaseMatcher（模板方法）
      registry.py               # @register
      config.py                 # RunConfig, build_matcher
      preprocess.py             # unproject, filter_by_mask, _to_gray_align32
      io.py                     # save_npz, result_to_manifest, snapshot_config
    methods/
      __init__.py               # 触发注册：from . import lightglue, efficient_loftr
      _thirdparty.py            # sys.path 注入 hook
      lightglue.py              # @register("lightglue")
      efficient_loftr.py        # @register("efficient_loftr")
    dataset/
      __init__.py
      protocol.py               # PairDataset Protocol
      from_pair_list.py         # 最小参考实现
      config.py                 # DatasetConfig + build_dataset
    cli/
      run.py                    # xmatcher run
  configs/
    lightglue.yaml
    efficient_loftr.yaml
    dataset/
      sample_pairs.yaml
  docker/
    Dockerfile
    requirements.common.txt
    requirements.lightglue.txt
    requirements.eloftr.txt
    build.sh
    .dockerignore
  scripts/
    download_weights.sh
    _download.py
  weights/
    README.md
    WEIGHTS.lock                # YAML 格式的权重清单（含 sha256）
  tests/
    unit/
    contract/
    smoke/
    fixtures/
      sample_a.jpg
      sample_b.jpg
      sample_pairs.txt
    conftest.py                 # gpu / requires_weights 标记
  thirdparty/
    LightGlue/                  # submodule
    EfficientLoFTR/             # submodule
    RoMaV2/                     # submodule（MVP 不接）
    accelerated_features/       # submodule（MVP 不接）
  .github/workflows/
    docker-build.yml
    test.yml
  pyproject.toml                # 库元信息 + 依赖（pydantic, pyyaml, opencv-python, gdown 等）
  README.md
```

### 2.3 数据流

```
Dataset(配置) ──► ImagePair(image0/1 + meta0/1 + pair_id + extras)
                                │
                                ▼
                       BaseMatcher.__call__
                       ┌─────────────────────────────┐
                       │ 1. _forward(pair)           │  子类实现：模型预处理 + 前向
                       │    → _RawOutput             │     输出 processed 坐标系点
                       │ 2. unproject(meta)          │  基类统一：→ 原图坐标
                       │ 3. filter_by_mask(meta)     │  基类统一：valid_mask 过滤
                       │ 4. 装包 MatchResult         │  添加 method/pair_id/runtime_ms
                       └─────────────────────────────┘
                                │
                                ▼
                       MatchResult(原图坐标 mkpts + mconf + 可选 dense)
                                │
                                ▼
            io.save_npz + manifest.jsonl + config.snapshot.yaml
```

## 3. 数据契约

### 3.1 PreprocessMeta

```python
@dataclass
class PreprocessMeta:
    original_size: tuple[int, int]       # (H_orig, W_orig)
    processed_size: tuple[int, int]      # (H_proc, W_proc)
    # —— 显式步骤字段（按 crop → scale → pad 的顺序应用）——
    crop_box: tuple[int, int, int, int] | None   # (x0, y0, x1, y1) on original
    scale: tuple[float, float]                   # (sx, sy)
    pad: tuple[int, int, int, int]               # (left, top, right, bottom)
    valid_mask: torch.Tensor | None              # (H_proc, W_proc) bool；None=全有效
    # —— 派生缓存（__post_init__ 算出，调用方不应手工赋值）——
    affine_proc_to_orig: torch.Tensor    # (2, 3) float
```

**变换语义**（`processed → original`）：

```
1. (u, v) on processed
2. 减去 pad.left / pad.top                 → (u', v') on processed-without-pad
3. 除以 scale.sx / scale.sy                → (u'', v'') on cropped original
4. 加上 crop_box.x0 / y0                   → (x, y) on original
```

`affine_proc_to_orig` 在 `__post_init__` 由 `crop_box / scale / pad` 解析出来，仅供 `unproject` 使用。显式三个字段保留供调试与可视化。

**约束**：

- `crop_box=None` 表示"没裁剪"，等价于 `(0, 0, W_orig, H_orig)`。
- `pad=(0,0,0,0)` 表示无 padding。
- `affine_proc_to_orig` 与显式字段必须一致，由 unit test 锁住（`test_preprocess_meta.py`）。

### 3.2 ImagePair

```python
@dataclass
class ImagePair:
    image0: torch.Tensor                  # (3, H_proc, W_proc), float [0, 1]
    image1: torch.Tensor
    meta0: PreprocessMeta
    meta1: PreprocessMeta
    pair_id: str                          # 例: "scene01__0042__0043"，用于落盘文件名
    extras: dict = field(default_factory=dict)   # GT/相机参数等透传
```

**约定**：

- `image0/1` 已是 dataset 预处理后的 RGB 张量。
- Matcher 不再做 crop/mask；可能做"模型预处理"（灰度、对齐 32 倍数）但其变换在适配器内部消化，不写进 `meta`。
- `extras` 是宽口袋通道，库代码不读不写；后续如 GT 评测要稳定 schema 再升级为明牌字段。

### 3.3 MatchResult

```python
@dataclass
class MatchResult:
    mkpts0: torch.Tensor                  # (K, 2) float，原图 0 上的 (u, v)
    mkpts1: torch.Tensor                  # (K, 2)
    mconf:  torch.Tensor                  # (K,) float，方法原生分数（不归一化）
    method: str                           # "lightglue" / "efficient_loftr"
    pair_id: str
    runtime_ms: float
    dense: DenseField | None = None       # 仅 RoMa 等方法填充

@dataclass
class DenseField:
    warp:      torch.Tensor               # 形状由方法约定
    certainty: torch.Tensor               # (H, W) float
    coord_space: Literal["processed", "original"]   # 通常 "processed"
```

**约定**：

- `mkpts0/1` 始终在**原图坐标系**（unproject 已应用，或 `return_processed_coords=True` 时为 processed 坐标系——由调用方明确选择）。
- `mconf` 不强制范围，文档为每个方法标注语义：
  - LightGlue：sigmoid 后的相似度，∈[0, 1]，越大越好。
  - EfficientLoFTR：coarse + fine 综合分数，∈[0, 1]，越大越好。
- 设备与 Matcher 一致；调用方按需 `.cpu()`。

### 3.4 PairDataset Protocol

```python
class PairDataset(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[ImagePair]: ...
```

库内仅提供 `FromPairListDataset`：读 `pairs.txt`（每行 `path0 path1`），按 YAML 配置生成 `ImagePair`。这是 CLI 跑 demo 用的最小实现，不是产品级 dataset。

## 4. Matcher 抽象

### 4.1 BaseMatcher

```python
class BaseMatcher(abc.ABC):
    method_name: ClassVar[str]            # 子类必填，由 @register 装饰器写入
    Params: ClassVar[type[BaseModel]]     # 子类指定的 pydantic 参数模型

    def __init__(self, *, device: str = "cuda", params: BaseModel):
        self.device = device
        self.params = params              # 已校验过的 ParamModel 实例
        self._setup()

    @abc.abstractmethod
    def _setup(self) -> None:
        """加载权重、构造网络。子类实现。"""

    @abc.abstractmethod
    def _forward(self, pair: ImagePair) -> "_RawOutput":
        """模型预处理 + 前向。返回 processed 坐标系下的点。"""

    @torch.inference_mode()
    def __call__(self, pair: ImagePair, *,
                 return_processed_coords: bool = False) -> MatchResult:
        t0 = time.perf_counter()
        raw = self._forward(pair)
        if return_processed_coords:
            mk0, mk1 = raw.mkpts0, raw.mkpts1
            keep = torch.ones(len(raw.mkpts0), dtype=torch.bool, device=raw.mkpts0.device)
        else:
            mk0 = unproject(raw.mkpts0, pair.meta0)
            mk1 = unproject(raw.mkpts1, pair.meta1)
            keep = filter_by_mask(mk0, mk1, pair.meta0, pair.meta1)
        return MatchResult(
            mkpts0=mk0[keep], mkpts1=mk1[keep], mconf=raw.mconf[keep],
            method=self.method_name,
            pair_id=pair.pair_id,
            runtime_ms=(time.perf_counter() - t0) * 1000,
            dense=raw.dense,
        )

@dataclass
class _RawOutput:
    """适配器内部输出。子类不直接构造 MatchResult。"""
    mkpts0: torch.Tensor                  # processed coords
    mkpts1: torch.Tensor
    mconf:  torch.Tensor
    dense:  DenseField | None
```

**关键性质**：

- 子类只关心"模型预处理 + 前向 + 出 processed 坐标"，**unproject 与 mask 过滤是基类统一做的**——保证所有方法行为一致。
- `return_processed_coords=True` 时同时跳过 unproject 与 mask 过滤——两者一起开关，逻辑统一。
- `_RawOutput` 私有，强制子类走基类装包路径。

### 4.2 注册表

```python
# xmatcher/core/registry.py
_REGISTRY: dict[str, type[BaseMatcher]] = {}

def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise KeyError(f"Matcher '{name}' already registered")
        cls.method_name = name
        _REGISTRY[name] = cls
        return cls
    return deco

def get_matcher_cls(name: str) -> type[BaseMatcher]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown matcher '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]
```

`xmatcher/methods/__init__.py` 显式 import 触发注册：

```python
from . import _thirdparty   # 必须先注入 sys.path
from . import lightglue, efficient_loftr
```

### 4.3 配置加载

`RunConfig`（外层）：

```python
class RunConfig(BaseModel):
    method: str
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    seed: int = 0
    params: dict                          # 内层，build_matcher 时按 method 路由
```

两段式校验：

```python
def build_matcher(cfg: RunConfig) -> BaseMatcher:
    cls = get_matcher_cls(cfg.method)
    typed_params = cls.Params(**cfg.params)
    set_seed(cfg.seed)
    return cls(device=cfg.device, params=typed_params)
```

外层校验定结构、内层校验定具体字段——错误信息精准到方法。

## 5. 方法适配器

### 5.1 LightGlueMatcher

```python
# xmatcher/methods/lightglue.py
class LightGlueParams(BaseModel):
    features: Literal["superpoint", "disk", "aliked", "sift"] = "superpoint"
    max_num_keypoints: int = 2048
    detection_threshold: float = 0.0005   # 喂 extractor
    match_threshold: float = 0.1          # 喂 LightGlue.filter_threshold

@register("lightglue")
class LightGlueMatcher(BaseMatcher):
    Params = LightGlueParams

    def _setup(self):
        from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT
        EXTRACTORS = {"superpoint": SuperPoint, "disk": DISK,
                      "aliked": ALIKED, "sift": SIFT}
        p = self.params
        self.extractor = EXTRACTORS[p.features](
            max_num_keypoints=p.max_num_keypoints,
            detection_threshold=p.detection_threshold,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features=p.features,
            filter_threshold=p.match_threshold,
        ).eval().to(self.device)

    def _forward(self, pair):
        from lightglue.utils import rbd
        img0 = pair.image0.to(self.device)
        img1 = pair.image1.to(self.device)
        # 关键：resize=None 关掉 extractor 内置 resize（dataset 已定型 H/W）
        feats0 = self.extractor.extract(img0, resize=None)
        feats1 = self.extractor.extract(img1, resize=None)
        out = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, out = [rbd(x) for x in [feats0, feats1, out]]
        m = out["matches"]
        return _RawOutput(
            mkpts0=feats0["keypoints"][m[:, 0]],
            mkpts1=feats1["keypoints"][m[:, 1]],
            mconf=out["scores"],
            dense=None,
        )
```

**风险点**：`extract(resize=None)` 必须显式关掉 LightGlue 内置 resize；否则 keypoints 落在它内部 grid 上，再 unproject 会错位。这是 sparse 适配器最容易出 bug 的地方，contract test 必须覆盖（用 mock + 不同 H/W 验证 unproject 端到端）。

权重：走 LightGlue 上游自动下载（`torch.hub`），不进我们的 `download_weights.sh`。

### 5.2 EfficientLoFTRMatcher

```python
# xmatcher/methods/efficient_loftr.py
class EfficientLoFTRParams(BaseModel):
    weights: Path                         # 必填，相对路径基于 XMATCHER_WEIGHTS_DIR
    precision: Literal["fp32", "fp16", "mp"] = "mp"
    match_threshold: float = 0.2
    border_rm: int = 2

    @field_validator("weights", mode="after")
    @classmethod
    def _resolve_weights(cls, v: Path) -> Path:
        if v.is_absolute():
            return v
        base = Path(os.environ.get("XMATCHER_WEIGHTS_DIR",
                                    Path.home() / ".cache" / "xmatcher"))
        resolved = base / v
        if not resolved.exists():
            raise FileNotFoundError(
                f"Weight not found: {resolved}\n"
                f"Run scripts/download_weights.sh efficient_loftr to fetch it."
            )
        return resolved

@register("efficient_loftr")
class EfficientLoFTRMatcher(BaseMatcher):
    Params = EfficientLoFTRParams

    def _setup(self):
        from src.loftr import LoFTR, reparameter
        from src.config.default import get_cfg_defaults
        cfg = get_cfg_defaults()
        cfg.LOFTR.MATCH_COARSE.THR = self.params.match_threshold
        cfg.LOFTR.MATCH_COARSE.BORDER_RM = self.params.border_rm
        self.matcher = LoFTR(config=cfg.LOFTR)
        state = torch.load(self.params.weights, map_location="cpu")
        self.matcher.load_state_dict(state["state_dict"])
        self.matcher = reparameter(self.matcher).eval().to(self.device)

    def _forward(self, pair):
        # 模型预处理：灰度 + 对齐 32 倍数（适配器内部，不进 meta）
        img0_pad, pad0 = _to_gray_align32(pair.image0)
        img1_pad, pad1 = _to_gray_align32(pair.image1)
        data = {
            "image0": img0_pad.to(self.device).unsqueeze(0),
            "image1": img1_pad.to(self.device).unsqueeze(0),
        }
        # precision 由 self.params.precision 决定 autocast 行为
        self.matcher(data)
        # 减去适配器内部 pad，回到 dataset 给的 processed 坐标系
        off0 = torch.tensor([pad0[0], pad0[1]], device=self.device, dtype=torch.float32)
        off1 = torch.tensor([pad1[0], pad1[1]], device=self.device, dtype=torch.float32)
        return _RawOutput(
            mkpts0=data["mkpts0_f"] - off0,
            mkpts1=data["mkpts1_f"] - off1,
            mconf=data["mconf"],
            dense=None,
        )
```

**风险点**：

- LoFTR 不支持 `pip install`，靠 `sys.path` 注入。`xmatcher/methods/_thirdparty.py` 把 `thirdparty/EfficientLoFTR` 加到 sys.path 顶部，使 `from src.loftr import ...` 解析到上游目录。
- 与 LightGlue 共用 sys.path 注入策略（即使 LightGlue 有 pyproject）——保持一致，少一道分叉。

### 5.3 共用工具

`xmatcher/core/preprocess.py`：

- `unproject(pts: Tensor, meta: PreprocessMeta) -> Tensor`：用 `meta.affine_proc_to_orig` 把点从 processed 映回 original。
- `filter_by_mask(p0, p1, m0, m1) -> Tensor[bool]`：双边都要落在 valid_mask 内（None 时跳过）；只有当 `m0/m1.valid_mask` 任一非 None 时才过滤。
- `_to_gray_align32(img: Tensor) -> tuple[Tensor, tuple[int, int]]`：RGB→灰度 + 右下 pad 到 32 倍数；返回 `(img_pad, (pad_left, pad_top))`。

### 5.4 sys.path 注入

```python
# xmatcher/methods/_thirdparty.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY = ROOT / "thirdparty"
for sub in ["LightGlue", "EfficientLoFTR"]:
    p = THIRDPARTY / sub
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

`methods/__init__.py` 第一行 `from . import _thirdparty`，确保两个适配器 import 时上游已可见。

## 6. 配置 / CLI / 入口

### 6.1 配置文件

method 配置（`configs/<name>.yaml`）：

```yaml
# configs/lightglue.yaml
method: lightglue
device: cuda
seed: 0
params:
  features: superpoint
  max_num_keypoints: 2048
  match_threshold: 0.1
```

```yaml
# configs/efficient_loftr.yaml
method: efficient_loftr
device: cuda
seed: 0
params:
  weights: efficient_loftr/eloftr_outdoor.ckpt   # 相对 XMATCHER_WEIGHTS_DIR
  precision: mp
  match_threshold: 0.2
```

dataset 配置（`configs/dataset/<name>.yaml`）：

```yaml
# configs/dataset/sample_pairs.yaml
type: from_pair_list
params:
  pairs_file: assets/sample_pairs.txt
  image_root: assets/images/
  preprocess:
    resize_long_side: 1024
    crop: null
    pad_to_multiple: null
```

method 与 dataset 配置分两份——dataset 复用率高，一份 dataset 配多个 method 是常态。

### 6.2 CLI

```bash
xmatcher run \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lightglue_sample/ \
    [--limit 5] [--device cuda] [--no-postprocess]
```

`--no-postprocess` ⇒ `BaseMatcher.__call__(return_processed_coords=True)`。

CLI 用 argparse + 单个 `run` 子命令；后续要加 `xmatcher visualize / xmatcher list-methods` 时再升级。

### 6.3 输出结构

```
outputs/lightglue_sample/
  manifest.jsonl                  # 每行: {pair_id, method, num_matches, runtime_ms, npz_path}
  matches/
    scene01__0042__0043.npz       # mkpts0, mkpts1, mconf, method, pair_id
  config.snapshot.yaml            # 合并后的运行配置 + git commit hash
```

`config.snapshot.yaml` 包含 `method_cfg`、`dataset_cfg`、`git_commit`、`xmatcher_version`、`run_timestamp`，作为复现这次运行的唯一可信源。

### 6.4 入口骨架

```python
# xmatcher/cli/run.py
def main():
    args = parse_args()
    method_cfg = RunConfig.model_validate(yaml.safe_load(open(args.method_cfg)))
    dataset_cfg = DatasetConfig.model_validate(yaml.safe_load(open(args.dataset_cfg)))
    if args.device:
        method_cfg.device = args.device

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matches").mkdir(exist_ok=True)
    snapshot_config(method_cfg, dataset_cfg, out / "config.snapshot.yaml")

    matcher = build_matcher(method_cfg)
    dataset = build_dataset(dataset_cfg)

    with (out / "manifest.jsonl").open("w") as mf:
        for i, pair in enumerate(dataset):
            if args.limit and i >= args.limit:
                break
            res = matcher(pair, return_processed_coords=args.no_postprocess)
            npz_path = out / "matches" / f"{pair.pair_id}.npz"
            save_npz(res, npz_path)
            mf.write(json.dumps(result_to_manifest(res, npz_path)) + "\n")
```

## 7. Docker 与构建管线

### 7.1 Dockerfile

```dockerfile
# docker/Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Layer 1: 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Layer 2: PyTorch
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Layer 3: 公共依赖
COPY docker/requirements.common.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.common.txt

# Layer 4: 各方法依赖
COPY docker/requirements.lightglue.txt docker/requirements.eloftr.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.lightglue.txt && \
    pip install --no-cache-dir -r /tmp/requirements.eloftr.txt

# Layer 5: 代码
WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app
ENV XMATCHER_WEIGHTS_DIR=/root/.cache/xmatcher

ENTRYPOINT ["python", "-m", "xmatcher.cli.run"]
CMD ["--help"]
```

`.dockerignore`：

```
.git
.github
.vscode
outputs/
weights/
**/__pycache__
**/*.pyc
.env
VRecHub
```

### 7.2 requirements 拆分

```
docker/
  requirements.common.txt        # numpy, opencv-python, pyyaml, pydantic, einops, gdown
  requirements.lightglue.txt     # kornia
  requirements.eloftr.txt        # pytorch-lightning, yacs, loguru, einops, h5py
```

按方法拆——加新方法只需新增 `requirements.<name>.txt` + Dockerfile 加一行。具体版本在实现期通过实测锁定。

### 7.3 本地构建脚本

```bash
# docker/build.sh
#!/bin/bash
set -e
git submodule update --init --recursive
docker build -t xmatcher:dev -f docker/Dockerfile .
```

本地 build 不需要 token——submodule 已在本机 checkout。`COPY . /app` 直接打包。

### 7.4 GitHub Actions

```yaml
# .github/workflows/docker-build.yml
name: Build & Push Image
on:
  push:
    branches: [main]
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Configure git for private submodules
        run: |
          git config --global url."https://${{ secrets.GH_PAT }}@github.com/".insteadOf "git@github.com:"
          git config --global url."https://${{ secrets.GH_PAT }}@github.com/".insteadOf "https://github.com/"
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GH_PAT }}
          submodules: recursive
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository_owner }}/xmatcher:latest
            ghcr.io/${{ github.repository_owner }}/xmatcher:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

GitHub repo Secrets 需要配置：

- `GH_PAT`：私有 submodule 访问 token（与 GHA 自动注入的 `GITHUB_TOKEN` 区分）。
- `GITHUB_TOKEN` 由 GHA 自动注入，用于推 GHCR。

**安全提示**：原始备忘 `docs/design/UNIFIED_INTERFACE_KPTS_MATCHING.md` 中曾贴明文 token；该 token 必须立即作废重新生成。新 token 仅放置于本地 `.env`（已 gitignore）和 GitHub Secrets。

### 7.5 用法

```bash
# 本地开发（覆盖代码）
docker run --rm --gpus all \
    -v $PWD:/app \
    -v $HOME/.cache/xmatcher:/root/.cache/xmatcher \
    xmatcher:dev \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg/

# 远程跑（自包含镜像）
docker run --rm --gpus all \
    -v $HOME/.cache/xmatcher:/root/.cache/xmatcher \
    -v $PWD/outputs:/app/outputs \
    ghcr.io/wangxinjian1108/xmatcher:latest \
    --method-cfg configs/lightglue.yaml \
    --dataset-cfg configs/dataset/sample_pairs.yaml \
    --out outputs/lg/
```

`--entrypoint bash` 覆盖默认 ENTRYPOINT 即可进入交互 shell。

## 8. 权重管理

### 8.1 原则

- 永远不进 git，永远不进 Docker 镜像。
- 统一缓存路径 `XMATCHER_WEIGHTS_DIR`（默认 `~/.cache/xmatcher/`）。
- 缺失时**硬错误** + 给出修复命令——不静默自动下载。
- LightGlue 权重走上游自动下载（`torch.hub`），不进我们的下载脚本。

### 8.2 目录与清单

```
~/.cache/xmatcher/                       # XMATCHER_WEIGHTS_DIR
  efficient_loftr/
    eloftr_outdoor.ckpt
    eloftr_indoor.ckpt
```

仓库内：

```
weights/
  README.md                              # 各权重的来源 + license
  WEIGHTS.lock                           # YAML 格式，含 url/sha256/target
```

`.gitignore` 中 `weights/*` 但保留 `weights/README.md` 与 `weights/WEIGHTS.lock`。

### 8.3 WEIGHTS.lock 示例

```yaml
efficient_loftr:
  outdoor:
    gdrive_id: "<填具体 ID>"
    sha256:    "<填具体值>"
    target:    "efficient_loftr/eloftr_outdoor.ckpt"
  indoor:
    gdrive_id: "<填具体 ID>"
    sha256:    "<填具体值>"
    target:    "efficient_loftr/eloftr_indoor.ckpt"
```

### 8.4 下载脚本

```bash
# scripts/download_weights.sh
#!/bin/bash
set -e
TARGET="${XMATCHER_WEIGHTS_DIR:-$HOME/.cache/xmatcher}"
mkdir -p "$TARGET"
METHODS="${1:-efficient_loftr}"
for m in $METHODS; do
    python scripts/_download.py --method "$m" --target "$TARGET"
done
```

`scripts/_download.py`：

- 解析 `weights/WEIGHTS.lock`。
- 对每个条目：检查目标文件存在 + sha256 匹配 → 跳过；否则下载。
- HTTP 用 `urllib`；Google Drive 用 `gdown`。
- 下载完做 sha256 校验，不匹配报错。

## 9. 测试

### 9.1 分层

```
tests/
  unit/                              # 不需要 GPU、不需要权重；CI 跑
    test_preprocess_meta.py
    test_unproject.py
    test_filter_by_mask.py
    test_registry.py
    test_config.py
    test_match_result.py
  contract/                          # 接口契约，用 MockMatcher；CI 跑
    test_base_matcher_template.py
    test_dataset_protocol.py
  smoke/                             # 跑真实方法；CI 不跑
    test_lightglue_smoke.py
    test_efficient_loftr_smoke.py
  fixtures/
    sample_a.jpg
    sample_b.jpg
    sample_pairs.txt
  conftest.py
```

### 9.2 关键 unit 测试要点

- `test_preprocess_meta.py`：构造 `PreprocessMeta` 时，`affine_proc_to_orig` 必须等于 `crop_box / scale / pad` 显式字段组合出的矩阵。已知点端到端验证（如 `(0,0)` proc → `(crop_x0, crop_y0)` original）。
- `test_unproject.py`：参数化覆盖 8 种组合（纯 resize、resize+crop、resize+pad、crop+resize+pad；左 pad / 右 pad 等）；用随机点 round-trip 验证（processed → original → processed 应回原点）。

### 9.3 关键 contract 测试要点

用 `MockMatcher`（`_forward` 返回固定的 keypoints）锁住模板逻辑：

- `test_base_call_applies_unproject`：默认 `__call__` 必须把 processed 点映回原图。
- `test_return_processed_coords_skips_unproject`：开关为 True 时不变换。
- `test_mask_filtering_drops_points_outside_valid_region`：`valid_mask` 之外的点被滤掉。

这一组测试是**库的契约护栏**，捕获"换适配器把模板逻辑写错"的回归。

### 9.4 Smoke 测试要点

```python
@pytest.mark.gpu
@pytest.mark.requires_weights("lightglue")
def test_lightglue_runs_on_sample_pair(sample_pair):
    matcher = build_matcher_from_yaml("configs/lightglue.yaml")
    res = matcher(sample_pair)
    assert res.method == "lightglue"
    assert res.mkpts0.shape == res.mkpts1.shape
    assert res.mkpts0.shape[1] == 2
    assert len(res.mkpts0) >= 50
    H, W = sample_pair.meta0.original_size
    assert (res.mkpts0[:, 0] >= 0).all() and (res.mkpts0[:, 0] < W).all()
    assert (res.mkpts0[:, 1] >= 0).all() and (res.mkpts0[:, 1] < H).all()
```

`@pytest.mark.gpu` / `@pytest.mark.requires_weights` 由 `conftest.py` 实现 skipif：无 CUDA 跳 gpu 标记；`XMATCHER_WEIGHTS_DIR` 缺权重跳 requires_weights 标记。本地 `pytest tests/` 自动只跑能跑的。

### 9.5 Fixtures

`tests/fixtures/sample_a.jpg / sample_b.jpg`：自带一份（不依赖 submodule），同场景两个视角，~512x768，附 LICENSE 备注。

### 9.6 CI

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
          token: ${{ secrets.GH_PAT }}
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e .[test]
      - run: pytest tests/unit tests/contract -v
```

CI 仅跑 unit + contract；smoke 留给本地或后续的 self-hosted runner。

## 10. 依赖与版本

| 名称 | 版本约束 | 备注 |
|------|----------|------|
| Python | 3.10 | Docker 锁定 3.10 |
| PyTorch | 2.4.1 + cu121 | RoMa v2 兼容性 |
| torchvision | 0.19.1 + cu121 |  |
| pydantic | >=2.0 | v2 API（field_validator 等） |
| pyyaml | * |  |
| numpy | * |  |
| opencv-python | * | dataset 预处理 |
| einops | * | 上游需要 |
| pytorch-lightning | 与 EfficientLoFTR 兼容 | 实现期实测锁定 |
| yacs | * | LoFTR 配置 |
| loguru | * | LoFTR |
| h5py | * | LoFTR 测试代码会用 |
| gdown | * | Google Drive 下载 |
| kornia | * | LightGlue 依赖 |

## 11. 风险与开放问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| LightGlue `extract` 内置 resize 未关掉 | 坐标错位，下游全错 | contract test 端到端验证；适配器代码注释强调 |
| LoFTR 权重 sha256 / Google Drive ID 当前空缺 | 下载脚本无法工作 | 实现阶段第一件事：下载一次官方权重，记录 sha256 与 ID 写入 `WEIGHTS.lock` |
| `pytorch-lightning` 版本与上游耦合 | `state_dict` 加载失败 | 实现期实测锁定一个版本，写入 `requirements.eloftr.txt` |
| 私有 submodule 在 GHA 中拉取 | CI 卡死 | `url.insteadOf` + `GH_PAT` 已规划 |
| GitHub PAT 已在备忘中泄露 | 安全风险 | **必须立即作废重新生成**；新 token 仅放本地 `.env` 与 repo Secrets |
| LoFTR 适配器内部 pad 偏移忘记减 | mkpts 错位 | contract test 用 mock + 非零 pad 验证 |

## 12. 后续工作（非 MVP）

1. 接入 RoMa v2（dense 透传通道首次实战）。
2. 接入 XFeat。
3. `xmatcher visualize` 子命令：读 manifest + npz，生成匹配可视化。
4. GT 评测：`extras` 升级为明牌 `GroundTruth` 字段；新增 `xmatcher eval` 子命令。
5. self-hosted GPU runner，开启 smoke 测试入 CI。
6. 权重也支持 HF Hub 镜像（中国大陆下载体验）。
