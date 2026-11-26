# FAQ

## HCCL error 7 / ERR02200：端口被其他训练进程占用

### 问题现象

分布式启动时所有 rank 报错：

```text
RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
HCCL function error: hcclCommInitRootInfoConfig(...) error code 7
DIST call hccl api failed
```

各用例报错**逐字一致**。穿透 `HCCLUtils.cpp` / `ERR02200` 表层，在 stderr 深层能找到关键一行：

```text
EI0020: Failed to enable listening ... The IP address 192.168.2.x and port 16666 have already been bound
```

### 问题根因

HCCL 默认 NPU socket 端口（`16666`，遵循 `HCCL_NPU_SOCKET_PORT_RANGE`）被机器上**其他用户的训练进程**占用。属于环境级问题，不是代码回归——判据是跨用例、跨分支报错逐字一致。

### 解决方案

1. 在 stderr 日志中搜索 `EI0020`，定位被占用的 IP/端口及占用进程（如 `torchrun`、`trainer.py`）；
2. 启动前设置 `HCCL_NPU_SOCKET_PORT_RANGE`，把端口范围错开到空闲段；
3. 或等待占用端口的训练结束后再启动。

## 升级 CANN / torch 后报 libstdc++ / CXXABI / GLIBCXX 符号缺失

### 问题现象

加载 torch_npu / TransformerEngineNPU 时运行期报：

```text
ImportError: .../libstdc++.so.6: version `CXXABI_1.3.15' not found
```

（同类还有 `GLIBCXX_3.4.xx not found`。）

### 问题根因

系统自带 libstdc++ 版本较旧（缺 `CXXABI_1.3.15`），而 conda 环境自带的 libstdc++ 够新，却默认不在动态库加载路径里 → 加载 torch_npu 时命中系统旧版。

> [!NOTE]
>
> 重建 conda 环境**不能**解决：新环境的 libstdc++ 版本相同，`conda activate` 依然不把自身 lib 加进加载路径，还是会命中系统旧版。

### 解决方案

让 conda 环境的 libstdc++ 进入加载路径，一行：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

写进 conda 环境的激活脚本 `$CONDA_PREFIX/etc/conda/activate.d/libstdcxx-fix.sh` 后，每次 `conda activate` 自动生效，不用手动执行。

## 昇腾设备初始化失败 / torch_npu owner 警告

### 问题现象

- NPU 初始化失败，或驱动/权限类报错；
- torch_npu 启动时打印 `owner` 相关 WARNING。

### 问题根因

当前用户不在 `HwHiAiUser` 组，无 NPU 设备访问权限。

### 解决方案

将训练用户加入 `HwHiAiUser` 组并重新登录：

```bash
sudo usermod -aG HwHiAiUser <训练用户>
```

`torch_npu` 的 owner WARNING 属于 cosmetic 提示，不影响运行，可忽略。

## 优化器报 Wrapped parameters must be one of torch.cuda.*Tensor

### 问题现象

模型构建阶段、优化器初始化时报错：

```text
TypeError: Wrapped parameters must be one of torch.cuda.FloatTensor,
torch.cuda.HalfTensor, or torch.cuda.BFloat16Tensor. Received torch.BFloat16Tensor
```

触发条件（需同时满足）：

- 使用了 `--use-cpu-initialization=True`，且未开启 `--use-torch-fsdp2`；
- 跑的是 Megatron-LM **`main` 分支**（含 ModelBuilder 集成）。

### 问题根因

Megatron-LM 当前 `main` 分支的模型构建存在两条并行的参数初始化路径，GPU 分配条件不一致：

| 路径 | 条件 | 语义 |
|------|------|------|
| 旧 `get_model()`（正确） | `not (use_torch_fsdp2 and use_cpu_initialization)` | 仅当两 flag 都为 True 才跳过 `.cuda()` |
| `builder.build_distributed_models()`（有 bug） | `not use_torch_fsdp2 and not use_cpu_initialization` | 任一 flag 为 True 就跳过 `.cuda()` |

`--use-cpu-initialization=True` 且默认 `--use-torch-fsdp2=False` 时，走 `builder.build_distributed_models()` 路径会跳过 `.cuda()` → 参数留在 CPU → 被包成 CPU bf16 参数 → 优化器类型校验失败。

### 解决方案

MA 已通过修复类功能提供处理：

- 在启动命令中加 **`--fix-use-cpu-initialization`**。该开关经 `MegatronFixFeature` 注册，
  开启后 `prepare_existing_model_chunks_for_distributed_training_wrapper` 会在调用原函数**之前**对模型做幂等
  `.cuda()`，恢复与旧 `get_model()` 一致的时序（不改上游条件，不影响其他调用方）。
- **无需处理的分支**：`core_r0.18.0` 等 release 分支没有 ModelBuilder 集成路径，buggy 代码不可达，无需该开关。

> **注意**：报错消失只是第一步，重点是训练早期指标——grad_norm 应为非零（如 42–60）、loss 正常下降。
> 若 grad_norm=0.000 且 loss 不收敛，说明设备迁移发生在 DDP wrap 之后、梯度 hook 失效（静默失败），需确认 MA 版本包含对应时序修正。

## 训练过程中迭代内显存不断持续上涨，迭代结束回落

### 问题现象

训练过程中，显存占用在每步迭代内都没有波动平面而是不断持续上涨，在迭代结束时回落至正常。此问题挤占 OOM 余量，影响可运行的模型规模与批大小。

### 问题根因

Megatron-LM core_r0.18.0分支默认在一次性侧流上完成 DDP/FSDP 包装初始化。部分场景下，受 PyTorch CachingAllocator 流内复用的约束影响，从主流向侧流持久性缓存的写入会使相关tensor被标记跨流依赖，对应显存块无法被主流复用。表现为训练过程中迭代内显存不断持续上涨，迭代结束时才被回收，显存回落的显存滞留现象，类似显存泄漏，部分情况可能会导致OOM。该问题已在 Megatron-LM 高版本中修复，core_r0.18.0 分支未包含对应修复。

### 解决方案

训练启动命令中追加 **`--fix-ddp-one-shot-stream`**（默认关闭）。启用后，DDP/FSDP 初始化与主流共用同一条流，消除上述显存滞留；修复方案与高版本 Megatron-LM 保持一致。

## 非 overlap 训练显存偏高（占位权重梯度常驻）

### 问题现象

开启梯度累加融合（gradient accumulation fusion，默认开启）后，显存中有一块与训练步数、微批次数无关的常驻占用：不随 `optimizer.zero_grad` 释放，也不随迭代回落，显存基线整体抬高，压缩 OOM 余量，部分场景直接 OOM。

### 问题根因

融合算子把权重梯度直接累加进 `main_grad`，autograd 返回的权重梯度（占位权重梯度，dummy wgrad）在数值上是废弃的。但它不能直接返回 `None`：当梯度规约与反向计算重叠（`--overlap-grad-reduce`）时，DDP 的后置 hook 依赖它触发，且 hook 内有 `param.grad is not None` 断言。因此 Megatron-LM core_r0.18.0 分支在开启融合时总是为每个权重申请一个与 `main_grad` 同形的占位张量；该张量在 TransformerEngineNPU 中以 `(shape, dtype, zero)` 为键缓存（`_dummy_wgrads`），跨层、跨迭代复用且永不释放，形成常驻占用。**未开启重叠（默认即未开启）时这份占位张量没有任何正确性价值**——此时梯度同步由 `finish_grad_sync` 显式整流 reduce 每个 bucket，不依赖逐参数的梯度就绪状态。

### 解决方案

训练启动命令中追加 **`--fix-non-overlap-dummy-wgrad`**（默认关闭）。启用后，非 overlap 场景下 Megatron 侧不再申请该占位张量；`--overlap-grad-reduce` 开启时行为与改动前逐位一致。

若开启后显存没有下降，按以下顺序排查：

1. 查看训练日志中的自检行 `[fix-non-overlap-dummy-wgrad] gated ...`：状态应为 `no dummy wgrad tensor allocated`；若为 `dummy wgrad still allocated (overlap path)`，说明当前开启了 `--overlap-grad-reduce`，本修复不适用。
2. TransformerEngine 自行分配占位张量的路径（如 MoE 的 GroupedLinear、量化分组 GEMM）需 TransformerEngineNPU 侧加入同一道闸门后才能受益，未适配时这部分仍保持原占用（不影响正确性）。
3. MindSpeed 等上层框架自带同构的占位张量分配代码（多为 `torch.empty` 逐次分配），不在本修复覆盖范围内。
