# FAQ

## HCCL Error 7/ERR02200: Port Occupied by Another Training Process

### Symptoms

At distributed startup, all ranks report the following error:

```text
RuntimeError: create_config:../torch_npu/csrc/distributed/HCCLUtils.cpp:140
HCCL function error: hcclCommInitRootInfoConfig(...) error code 7
DIST call hccl api failed
```

The error is **identical, word for word**, in every test case. Beyond the surface layer of `HCCLUtils.cpp`/`ERR02200`, a deeper scan of stderr reveals one key line:

```text
EI0020: Failed to enable listening ... The IP address 192.168.2.x and port 16666 have already been bound
```

### Root Cause

The default HCCL NPU socket port (16666, which follows `HCCL_NPU_SOCKET_PORT_RANGE`) is occupied by the **training processes of other users** on the machine. This is an environment-level problem, not a code regression. The diagnostic criterion is an error that is identical, word for word, across test cases and branches.

### Solution

1. Search the stderr log for `EI0020` to locate the occupied IP address and port as well as the process that occupies them (for example, `torchrun` or `trainer.py`).
2. Before startup, set `HCCL_NPU_SOCKET_PORT_RANGE` to shift the port range to a free segment.
3. Alternatively, wait until the training that occupies the port finishes, and then start.

## Missing libstdc++/CXXABI/GLIBCXX Symbols After You Upgrade CANN/torch

### Symptoms

At runtime, loading `torch_npu`/TransformerEngineNPU reports the following error:

```text
ImportError: .../libstdc++.so.6: version `CXXABI_1.3.15' not found
```

(A similar error is `GLIBCXX_3.4.xx not found`.)

### Root Cause

The libstdc++ that ships with the system is older (it lacks `CXXABI_1.3.15`), while the libstdc++ that ships with the conda environment is new enough. However, the latter is not in the dynamic library load path by default. Therefore, loading `torch_npu` picks up the old system version.

> [!NOTE]
>
> Rebuilding the conda environment does **not** help. The libstdc++ version in the new environment is the same, and `conda activate` still does not add its own `lib` to the load path. Therefore, the old system version is still picked up.

### Solution

To put the libstdc++ of the conda environment into the load path, use one line:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

After you write it into the activation script of the conda environment, `$CONDA_PREFIX/etc/conda/activate.d/libstdcxx-fix.sh`, it takes effect automatically on every `conda activate` and you do not need to run it manually.

## Ascend Device Initialization Failure/torch_npu owner Warning

### Symptoms

- NPU initialization fails, or a driver- or permission-related error is reported.
- `torch_npu` prints an `owner`-related WARNING at startup.

### Root Cause

The current user is not in the `HwHiAiUser` group and therefore has no access permission for NPU devices.

### Solution

Add the training user to the `HwHiAiUser` group and log in again:

```bash
sudo usermod -aG HwHiAiUser <training user>
```

The owner WARNING of `torch_npu` is a cosmetic message. It does not affect operation, and you can ignore it.

## Optimizer Reports Wrapped parameters must be one of torch.cuda.*Tensor

### Symptoms

The error occurs during model construction, at optimizer initialization:

```text
TypeError: Wrapped parameters must be one of torch.cuda.FloatTensor,
torch.cuda.HalfTensor, or torch.cuda.BFloat16Tensor. Received torch.BFloat16Tensor
```

Trigger conditions (all of them must be met):

- You use `--use-cpu-initialization=True` and you do not enable `--use-torch-fsdp2`.
- You run the **`main` branch** of Megatron-LM (which includes the ModelBuilder integration).

### Root Cause

The current `main` branch of Megatron-LM has two parallel parameter initialization paths during model construction, and their conditions for GPU allocation are inconsistent:

| Path | Condition | Semantics |
|------|-----------|-----------|
| Legacy `get_model()` (correct) | `not (use_torch_fsdp2 and use_cpu_initialization)` | Skips `.cuda()` only when both flags are True |
| `builder.build_distributed_models()` (buggy) | `not use_torch_fsdp2 and not use_cpu_initialization` | Skips `.cuda()` as soon as either flag is True |

When `--use-cpu-initialization=True` and the default `--use-torch-fsdp2=False` apply, the `builder.build_distributed_models()` path skips `.cuda()`. Therefore, the parameters remain on the CPU, are wrapped as CPU bf16 parameters, and fail the optimizer type check.

### Solution

MA already provides a fix feature to handle this case:

- Add **`--fix-use-cpu-initialization`** to the startup command. This switch is registered by `MegatronFixFeature`.
  After you enable it, `prepare_existing_model_chunks_for_distributed_training_wrapper` performs an idempotent
  `.cuda()` on the model **before** it calls the original function, which restores the same ordering as the legacy `get_model()` (it does not change the upstream condition and does not affect other callers).
- **Branches that need no action**: release branches such as `core_r0.18.0` have no ModelBuilder integration path. Therefore, the buggy code is unreachable and the switch is not needed.

> **Note**: The error disappearing is only the first step. The focus is the early-stage training metrics. `grad_norm` should be non-zero (for example, 42–60), and loss should decrease normally.
> If `grad_norm=0.000` and loss does not converge, this indicates that the device migration happens after the DDP wrap and that the gradient hook stops working (a silent failure). In this case, confirm that your MA version includes the corresponding ordering fix.

## Memory Usage Keeps Rising Within an Iteration During Training and Falls Back When the Iteration Ends

### Symptoms

During training, memory usage does not show a flat plateau within each step of an iteration. Instead, it keeps rising continuously and falls back to normal when the iteration ends. This problem uses up the OOM headroom and affects the model size and batch size that can run.

### Root Cause

The Megatron-LM core_r0.18.0 branch performs DDP/FSDP wrapper initialization on a one-shot side stream by default. In some scenarios, constrained by the in-stream reuse of the PyTorch CachingAllocator, writes from the main stream into the persistent cache of the side stream cause the related tensors to be marked as having a cross-stream dependency. Therefore, the corresponding memory block cannot be reused by the main stream. As a result, memory usage keeps rising within iterations during training and is reclaimed only when the iteration ends. This is a memory retention phenomenon that resembles a memory leak and may cause an OOM in some cases. This problem is fixed in later Megatron-LM versions, and the core_r0.18.0 branch does not include the corresponding fix.

### Solution

Append **`--fix-ddp-one-shot-stream`** to the training startup command (disabled by default). After you enable it, DDP/FSDP initialization shares the same stream as the main stream, which eliminates the aforementioned memory retention. This fix is consistent with the one in later Megatron-LM versions.

## High Memory Usage in Non-Overlap Training (Placeholder Weight Gradients Stay Resident)

### Symptoms

After you enable gradient accumulation fusion (it is enabled by default), a resident allocation appears in memory that is unrelated to the number of training steps and micro-batches. It is not released by `optimizer.zero_grad` and does not fall back as iterations proceed. Therefore, the memory baseline is raised overall, the OOM headroom is reduced, and in some scenarios the process runs directly out of memory.

### Root Cause

The fused operator accumulates weight gradients directly into `main_grad`, and the weight gradient returned by autograd (a placeholder weight gradient, or dummy wgrad) is numerically discarded. However, it cannot return `None` directly. When gradient reduction overlaps with the backward computation (`--overlap-grad-reduce`), the post hook of DDP relies on this gradient to fire, and the hook contains the `param.grad is not None` assertion. Therefore, when fusion is enabled, the Megatron-LM core_r0.18.0 branch always allocates a placeholder tensor with the same shape as `main_grad` for each weight. TransformerEngineNPU caches this tensor with `(shape, dtype, zero)` as the key (`_dummy_wgrads`), and the tensor is reused across layers and iterations and is never released, which results in a resident allocation. **When overlap is not enabled (it is not enabled by default), this placeholder tensor has no correctness value.** In this case, gradient synchronization reduces each bucket explicitly through `finish_grad_sync` and does not depend on the per-parameter gradient readiness state.

### Solution

Append **`--fix-non-overlap-dummy-wgrad`** to the training startup command (disabled by default). After you enable it, the Megatron side no longer allocates this placeholder tensor in non-overlap scenarios. When `--overlap-grad-reduce` is enabled, the behavior is bit-for-bit identical to that before the change.

If memory usage does not decrease after you enable it, troubleshoot in the following order:

1. Check the self-check line `[fix-non-overlap-dummy-wgrad] gated ...` in the training log. The status should be `no dummy wgrad tensor allocated`. If the status is `dummy wgrad still allocated (overlap path)`, `--overlap-grad-reduce` is currently enabled and this fix does not apply.
2. Paths where TransformerEngine allocates the placeholder tensor by itself (for example, GroupedLinear of MoE and quantized grouped GEMM) benefit only after the same gate is added on the TransformerEngineNPU side. Before that adaptation, these paths still keep the original allocation (correctness is not affected).
3. Upper-layer frameworks such as MindSpeed have their own equivalent placeholder tensor allocation code (mostly per-call allocation through `torch.empty`), which is outside the scope of this fix.
