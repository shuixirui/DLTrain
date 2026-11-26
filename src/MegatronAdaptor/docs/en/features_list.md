# Adaptor Interface List

## Usage Notes

- **Directory navigation**: This document is organized by feature category. You can jump to the corresponding parameter table directly from the table of contents on the left or from a section heading.
- **Feature enablement**: Parameters marked "effective only with `--optimization-level >= N`" belong to MegatronAdaptor feature capabilities (for example, NPU Profiling, NPU DataDump, and NPU deterministic computation). When the optimization level is not reached, you can still pass the parameter, but it does not take effect.
- **Troubleshooting parameters that do not take effect**: First confirm whether `--optimization-level` meets the requirement, then confirm whether you have run `import megatron_adaptor`. If the parameter still does not take effect, see the Troubleshooting section of the [Quick Start](quick_start.md).
- **Notation conventions**: "Boolean switch" means a true/false switch. "Megatron native parameter" means a parameter defined by Megatron-LM and passed through or redirected by MegatronAdaptor.

## Activation Functions

| Parameter Name | Description | Constraint |
|--------|------|------|
| --swiglu | Use the SwiGLU activation function. | Mutually exclusive with `--squared-relu` and `--quick-geglu`. |
| --squared-relu | Use the Squared ReLU activation function. | Mutually exclusive with `--swiglu`. |
| --openai-gelu | Use the OpenAI approximate GELU activation function. | Legacy parameter with no mutual exclusion validation. |
| --quick-geglu | Use the fast GeGLU activation function. | Mutually exclusive with `--swiglu`. |
| --use-te-activation-func | Use the activation functions implemented by Transformer Engine. | Boolean switch. Mutually exclusive with `bias_activation_fusion`. Supports only `gelu`, `silu`, and `relu`. |
| --use-fused-weighted-squared-relu | Enable the fused weighted Squared ReLU kernel. | Boolean switch. Used for the fused weighted Squared ReLU kernel in MoE scenarios. |
| --no-bias-gelu-fusion | Disable the bias-GELU fusion kernel. | Boolean switch. Automatically disabled when `add_bias_linear=False`. |
| --no-bias-swiglu-fusion | Disable the bias-SwiGLU fusion kernel. | Boolean switch. Takes effect only for `--swiglu`. |

## Attention Mechanisms

| Parameter Name | Description | Constraint |
|--------|------|------|
| --use-flash-attn | Enable Flash Attention to accelerate attention computation. | Boolean switch. |
| --flash-decode | Enable Flash decode mode. | Boolean switch. Cannot be used together with `--no-rope-freq`. |
| --group-query-attention | Enable Group Query Attention (GQA), where multiple Q heads share KV heads. | Boolean switch. Use it with `--num-query-groups` to set the number of groups. Mutually exclusive with `--multi-latent-attention`. |
| --multi-latent-attention | Enable Multi-Latent Attention (MLA), which compresses KV with a low rank to reduce memory usage. | Boolean switch. Use it with `--q-lora-rank`, `--kv-lora-rank`, `--qk-head-dim`, and `--v-head-dim`. Mutually exclusive with `--group-query-attention`. |
| --q-lora-rank | Low-rank compression dimension of the Q matrix in MLA. | int, default None (defaults to 512 in MLA configurations). Requires `--multi-latent-attention`. |
| --kv-lora-rank | Low-rank compression dimension of the KV matrix in MLA. | int, default 32 (defaults to 512 in MLA configurations). Requires `--multi-latent-attention`. |
| --qk-head-dim | QK dimension of each head in MLA. | int, default 128. Requires `--multi-latent-attention`. |
| --qk-pos-emb-head-dim | Head dimension of the positional embedding part in MLA. | int, default 64. Requires `--multi-latent-attention`. |
| --v-head-dim | V dimension of each head in MLA. | int, default 128. Requires `--multi-latent-attention`. |
| --qk-layernorm | Apply LayerNorm to the Q and K vectors separately. | Boolean switch. |
| --qk-l2-norm | Apply L2 normalization to the Q and K vectors. | Boolean switch (Llama4 style). |
| --relative-attention-num-buckets | Set the number of buckets for the T5-style relative attention bias. | int, default 32. Valid only with `--position-embedding-type relative`. |
| --relative-attention-max-distance | Set the maximum distance of the relative attention bias. | int, default 128. Requires `--position-embedding-type relative`. |
| --apply-query-key-layer-scaling | Scale attention scores by dividing them by √`head_dim`. | Boolean switch. Automatically sets `--attention-softmax-in-fp32` when enabled. |
| --attention-backend | Specify the backend implementation for attention computation. | Options: `auto`, `flash`, `fused`, `unfused`, `local`. Default: `auto`. |
| --attention-output-gate | Apply a gating mechanism to the attention output. | Boolean switch. |
| --window-size | Set the window size for sliding window attention. | tuple(int,int), default None (global attention). -1 means an infinite window. |
| --window-attn-skip-freq | Skip frequency of window attention, which uses global attention every N layers. | int. Requires `--window-size`. |
| --disable-bias-linear | Disable the bias in all linear layers. | Boolean switch (sets `add_bias_linear` to False). |
| --add-qkv-bias | Add bias only to the QKV projection layers. | Boolean switch, default False. Automatically set to True when `add_bias_linear=True`. |

## Position Embeddings

| Parameter Name | Description | Constraint |
|--------|------|------|
| --position-embedding-type | Position embedding type. | Options: `learned_absolute`, `rope`, `yarn`, `mrope`, `relative`, `none`. Default: `learned_absolute`. |
| --use-rotary-position-embeddings | Enable rotary position embeddings (RoPE). | Boolean switch (deprecated). When set, forces `position_embedding_type='rope'`. |
| --rope-type | Specific implementation type of RoPE. | Options: `rope`, `yarn`. Default: None (defaults to `yarn` for MLA and `rope` for regular attention). |
| --rotary-base | RoPE frequency base θ. | int, default 10000. |
| --rotary-percent | Apply rotary encoding to the first N% of the dimensions of each head. | float, default 1.0 (all dimensions). |
| --rotary-seq-len-interpolation-factor | RoPE sequence length interpolation factor, which extrapolates longer sequences. | float, default None. |
| --use-rope-scaling | Enable RoPE scaling (linear extrapolation). | Boolean switch. Use it with `--rope-scaling-factor`. |
| --rope-scaling-factor | RoPE linear scaling factor. | float, default 8.0. Requires `--use-rope-scaling`. |
| --no-rope-freq | Specify the frequency of layers that skip RoPE (no RoPE every N layers). | int or list, default None. Cannot be used together with `--flash-decode`. When it is an int, it must divide `num_layers` evenly. |
| --rotary-interleaved | Apply rotary dimensions in an interleaved manner instead of splitting them in half. | Boolean switch. Cannot be used together with `--multi-latent-attention`. |
| --no-rope-fusion | Disable the fused kernel optimization of RoPE. | Boolean switch. |
| --rotary-scaling-factor | Rotary scaling factor. | float, default 1.0 (defaults to 40 in MLA configurations). |
| --mscale | mscale parameter in YaRN, which controls attention temperature. | float, default 1.0. |
| --mscale-all-dim | Apply mscale to all dimensions in YaRN. | float, default 0.0. |
| --mrope-section | Specify the dimension sectioning of multi-dimensional RoPE (M-RoPE). | list of int. Required when `--position-embedding-type mrope`. |
| --max-position-embeddings | Maximum position embedding length that the model supports. | int, default None. Must be >= `seq_length`. |

## Normalization

| Parameter Name | Description | Constraint |
|--------|------|------|
| --normalization | Normalization layer type. | Options: `RMSNorm`, `LayerNorm`. Default: `LayerNorm`. |
| --norm-epsilon | Epsilon in normalization computation, which prevents division by zero. | float, default 1e-5. |
| --apply-residual-connection-post-layernorm | Add the residual connection after LayerNorm instead of before it. | Boolean switch. |
| --no-persist-layer-norm | Disable persistent LayerNorm. | Boolean switch. |
| --apply-wd-to-qk-layernorm | Apply weight decay to the parameters of QK LayerNorm as well. | Boolean switch. |
| --fused-residual-rmsnorm | Use the fused RMSNorm plus residual connection kernel. | Boolean switch. Requires `--normalization RMSNorm`. |

## Mixed Precision Training

| Parameter Name | Description | Constraint |
|--------|------|------|
| --bf16 | Use BF16 mixed precision training. | Boolean switch. Mutually exclusive with `--fp16`. |
| --fp16 | Use FP16 mixed precision training. | Boolean switch. Mutually exclusive with `--bf16`. |
| --fp32-residual-connection | Use FP32 precision for residual connections (improves numerical stability). | Boolean switch. |
| --fp8-format | FP8 low-precision quantization format. | Options: `e4m3`, `hybrid`. Requires the TE library. |
| --fp8-recipe | FP8 quantization strategy. | Options: `tensorwise`, `delayed`, `mxfp8`, `blockwise`. Requires `--fp8-format`. |
| --fp8-margin | Margin value for FP8 amax computation. | int, default 0. |
| --fp8-interval | Update frequency of the FP8 scaling factor (updated every N steps). | int, default 1. |
| --fp8-amax-history-len | Window length of the FP8 amax history. | int, default 1. |
| --fp8-amax-compute-algo | FP8 amax computation strategy. | Options: `most_recent`, `max`. |
| --no-fp8-wgrad | Disable FP8 weight gradient computation. | Boolean switch. |
| --fp8-param-gather | Use FP8 communication during parameter AllGather. | Boolean switch. Requires `--fp8-format`. |
| --fp8-quantizer-factory | Python implementation path of a custom FP8 quantizer. | str. Used for the `custom` quantization strategy. |
| --reuse-grad-buf-for-mxfp8-param-ag | Reuse the gradient buffer during MXFP8 parameter gather to save memory. | Boolean switch. Requires `--fp8-recipe mxfp8`. |
| --keep-fp8-transpose-cache | Keep the FP8 transpose cache. | Boolean switch. |
| --te-precision-config-file | Path of the Transformer Engine precision configuration file. | str (YAML/JSON). |
| --first-last-layers-bf16 | Use BF16 for the first and last layers (the remaining layers can use FP8). | Boolean switch. |
| --num-layers-at-start-in-bf16 | Number of layers that use BF16 at the start. | int. Requires `--first-last-layers-bf16`. |
| --num-layers-at-end-in-bf16 | Number of layers that use BF16 at the end. | int. Requires `--first-last-layers-bf16`. |
| --grad-reduce-in-bf16 | Use BF16 communication for gradient AllReduce. | Boolean switch. |
| --disable-bf16-reduced-precision-matmul | Disable BF16 low-precision matrix multiplication accumulation. | Boolean switch. |
| --attention-softmax-in-fp32 | Compute attention softmax in FP32. | Boolean switch. Automatically enabled when `--apply-query-key-layer-scaling` is enabled. |
| --fp16-lm-cross-entropy | Compute language model cross entropy in FP16. | Boolean switch. Requires `--fp16`. |
| --loss-scale | Fixed loss scale value (no dynamic scaling). | float, default None (uses dynamic scaling). |
| --initial-loss-scale | Initial value of dynamic loss scaling. | float, default 2^32. |
| --loss-scale-window | Loss scale window size (doubles after N consecutive steps without overflow). | int, default 1000. |
| --min-loss-scale | Minimum loss scale value. | float, default 1.0. |
| --hysteresis | Loss scale hysteresis parameter (allows N consecutive overflows before it reduces the scale). | int, default 2. |
| --accumulate-allreduce-grads-in-fp32 | Accumulate gradient AllReduce in FP32. | Boolean switch. |

## Parallelism

### Tensor Parallelism

| Parameter Name | Description | Constraint |
|--------|------|------|
| --tensor-model-parallel-size | Tensor parallelism degree, which splits model weights by column or row across multiple GPUs. | int, default 1. |
| --tp-comm-overlap | Enable overlap of TP communication and computation (AllGather/ReduceScatter). | Boolean switch, default False. The following disable parameters take effect only after you enable it. |
| --tp-comm-overlap-cfg | Path of the YAML file that specifies the TP communication overlap configuration. | str (YAML). Requires `--tp-comm-overlap`. |
| --disable-tp-comm-bulk-wgrad | Disable the bulk wgrad overlap of TP communication. | Boolean switch. Requires `--tp-comm-overlap`. |
| --disable-tp-comm-bulk-dgrad | Disable the bulk dgrad overlap of TP communication. | Boolean switch. Requires `--tp-comm-overlap`. |
| --disable-tp-comm-overlap-ag | Disable the AllGather overlap of TP communication. | Boolean switch. Requires `--tp-comm-overlap`. |
| --disable-tp-comm-overlap-rs | Disable the ReduceScatter overlap of TP communication. | Boolean switch. Requires `--tp-comm-overlap`. |
| --disable-tp-comm-split-ag | Disable Split AllGather in TP communication. | Boolean switch. Requires `--tp-comm-overlap`. |
| --disable-tp-comm-split-rs | Disable Split ReduceScatter in TP communication. | Boolean switch. Requires `--tp-comm-overlap`. |
| --no-clone-scatter-output-in-embedding | Disable cloning of the scatter output in the embedding. | Boolean switch. |

### Pipeline Parallelism

| Parameter Name | Description | Constraint |
|--------|------|------|
| --pipeline-model-parallel-size | Pipeline parallelism degree, which splits the model by layer across multiple stages. | int, default 1. |
| --pipeline-model-parallel-layout | Custom pipeline parallel layout (layer assignment for each stage). | str. Mutually exclusive with `--decoder-first/last-pipeline-num-layers`. |
| --num-layers-per-virtual-pipeline-stage | Number of layers per virtual pipeline stage (enables VPP). | int. It must be divisible by the total number of layers. |
| --num-virtual-stages-per-pipeline-rank | Number of virtual stages per pipeline rank. | int. Choose this parameter or `--num-layers-per-virtual-pipeline-stage`, not both. |
| --decoder-first-pipeline-num-layers | Number of layers in the first pipeline stage of the decoder. | int. Mutually exclusive with `--pipeline-model-parallel-layout`. |
| --decoder-last-pipeline-num-layers | Number of layers in the last pipeline stage of the decoder. | int. Mutually exclusive with `--pipeline-model-parallel-layout`. |
| --account-for-embedding-in-pipeline-split | Count the embedding layer in the layer count when the pipeline is split. | Boolean switch. |
| --account-for-loss-in-pipeline-split | Count the loss layer in the layer count when the pipeline is split. | Boolean switch. |
| --microbatch-group-size-per-virtual-pipeline-stage | Number of micro-batches executed consecutively per virtual stage in VPP. | int, default 1. Requires VPP to be enabled. |
| --pipeline-model-parallel-comm-backend | Communication backend for pipeline parallelism. | Options: `nccl`, `ucc`. |
| --overlap-p2p-communication-warmup-flush | Overlap the warmup flush phase of P2P communication. | Boolean switch. |
| --no-overlap-p2p-communication | Disable P2P communication overlap. | Boolean switch. Mutually exclusive with `--overlap-p2p-communication-warmup-flush`. |

### Sequence Parallelism

| Parameter Name | Description | Constraint |
|--------|------|------|
| --sequence-parallel | Enable sequence parallelism, which splits LayerNorm/Dropout along the sequence dimension. | Boolean switch. Requires `--tensor-model-parallel-size > 1`. |

### Context Parallelism

| Parameter Name | Description | Constraint |
|--------|------|------|
| --context-parallel-size | Context parallelism degree, which splits long sequences across multiple GPUs. | int, default 1. |
| --cp-comm-type | Communication type for context parallelism. | Options: `p2p`, `a2a`, `a2a+p2p`, `all_gather`. |
| --hierarchical-context-parallel-sizes | Size of each level in hierarchical CP. | list of int. |
| --max-seqlen-per-dp-cp-rank | Upper limit of the maximum sequence length of the local batch for each (DP rank, CP rank). | int. Used in variable-length sequence scenarios. |

### Expert Parallelism

| Parameter Name | Description | Constraint |
|--------|------|------|
| --expert-model-parallel-size | Expert parallelism degree, which distributes MoE experts across multiple GPUs. | int, default 1. Requires `--num-experts`. |
| --expert-tensor-parallel-size | Expert tensor parallelism degree. | int, defaults to the value of `--tensor-model-parallel-size`. |
| --overlap-moe-expert-parallel-comm | Overlap MoE expert parallel communication with computation. | Boolean switch. |
| --ep-overlap-early-attn-memory-release | Release attention memory earlier during EP overlap. | Boolean switch. Requires `--overlap-moe-expert-parallel-comm`. |

### Data Parallelism

| Parameter Name | Description | Constraint |
|--------|------|------|
| --data-parallel-sharding-strategy | Data parallel sharding strategy. | Options: `no_shard`, `optim`, `optim_grads`, `optim_grads_params`. Default: `optim_grads_params`. |
| --outer-dp-sharding-strategy | Outer data parallel sharding strategy. | Options: `no_shard`, `optim`. Default: `no_shard`. |
| --data-parallel-random-init | Data parallel random initialization. | Boolean switch. |
| --ddp-num-buckets | Number of DDP buckets. | int, default None (automatic). |
| --ddp-bucket-size | DDP bucket size. | int, default None (automatic). |
| --ddp-pad-buckets-for-high-nccl-busbw | Pad buckets for high NCCL bandwidth. | Boolean switch. |
| --ddp-reduce-scatter-with-fp32-accumulation | Use FP32 accumulation for DDP ReduceScatter. | Boolean switch. |
| --ddp-average-in-collective | Average in the collective operation. | Boolean switch. |

## Fully Sharded Data Parallel (FSDP)

| Parameter Name | Description | Constraint |
|--------|------|------|
| --use-megatron-fsdp | Enable Megatron FSDP fully sharded data parallel. | Boolean switch. |
| --fsdp-double-buffer | Enable FSDP double buffering (prefetch the parameters of the next round). | Boolean switch. Requires `--use-megatron-fsdp`. |
| --torch-fsdp2-no-reshard-after-forward | Do not reshard after the forward pass in FSDP2 (similar to ZeRO-2). | Boolean switch. |

## Optimizers

| Parameter Name | Description | Constraint |
|--------|------|------|
| --optimizer | Optimizer type. | Options: `adam`, `sgd`, `muon`, `dist_muon`, `lion`, `soap`, `adaptive_muon`. Default: `adam`. |
| --adam-beta1 | Adam first-order momentum decay rate. | float, default 0.9. |
| --adam-beta2 | Adam second-order momentum decay rate. | float, default 0.999. |
| --adam-eps | Adam epsilon, which prevents division by zero. | float, default 1e-8. |
| --sgd-momentum | SGD momentum coefficient. | float, default 0.9. Requires `--optimizer sgd`. |
| --muon-momentum | Muon momentum coefficient. | float, default 0.9. Requires the `muon` or `dist_muon` value of `--optimizer`. |
| --muon-no-split-qkv | Do not split QKV weights in Muon (optimize them as a whole). | Boolean switch (splits by default). Requires the Muon optimizer. |
| --muon-nesterov | Enable Nesterov momentum in Muon. | Boolean switch. Requires the Muon optimizer. |
| --muon-scale-mode | Muon scaling mode. | Options: `spectral`, `unit_rms_norm`, `shape_scaling`. Default: `spectral`. Requires the Muon optimizer. |
| --muon-fp32-matmul-prec | Muon FP32 matrix multiplication precision. | Options: `low`, `medium`, `high`. Default: `medium`. Requires the Muon optimizer. |
| --muon-coefficient-type | Newton-Schulz coefficient type of Muon. | str. Requires the Muon optimizer. |
| --muon-num-ns-steps | Number of Newton-Schulz iterations in Muon. | int, default 5. Requires the Muon optimizer. |
| --muon-tp-mode | Muon TP mode. | Options: `blockwise`, `duplicated`, `distributed`. Default: `blockwise`. Requires the Muon optimizer. |
| --muon-extra-scale-factor | Extra scaling factor of Muon. | float, default 1.0. Requires the Muon optimizer. |
| --muon-scalar-optimizer | Optimizer used for Muon scalar parameters. | Options: `adam`, `lion`. Default: `adam`. Requires the Muon optimizer. |
| --lion-beta1 | Lion optimizer first-order momentum. | float, default 0.95. Requires `--optimizer lion`. |
| --lion-beta2 | Lion optimizer second-order momentum. | float, default 0.98. Requires `--optimizer lion`. |
| --use-distributed-optimizer | Enable the distributed optimizer (ZeRO-1 style). | Boolean switch. |
| --num-distributed-optimizer-instances | Number of distributed optimizer instances. | int, default 1. Requires `--use-distributed-optimizer`. |
| --distrib-optim-fully-reshardable-mem-efficient | Enable the fully reshardable distributed optimizer (saves memory). | Boolean switch. Requires `--use-distributed-optimizer`. |
| --override-opt-param-scheduler | Override the optimizer parameter scheduler (use the CLI values). | Boolean switch. Mutually exclusive with `--use-checkpoint-opt-param-scheduler`. |
| --use-checkpoint-opt-param-scheduler | Use the optimizer parameter scheduler saved in the checkpoint. | Boolean switch. Mutually exclusive with `--override-opt-param-scheduler`. |

## Learning Rate Scheduling

| Parameter Name | Description | Constraint |
|--------|------|------|
| --lr | Initial learning rate. | float, required. |
| --min-lr | Minimum learning rate (lower bound of decay). | float, default 0.0. |
| --decoupled-lr | Decoupled learning rate (used in µP and similar scenarios). | float, default None. |
| --decoupled-min-lr | Decoupled minimum learning rate. | float, default None. |
| --lr-decay-style | Learning rate decay strategy. | Options: `constant`, `linear`, `cosine`, `inverse-square-root`, `WSD`. Default: `linear`. |
| --lr-decay-iters | Number of iterations for learning rate decay. | int. Choose this parameter or `--lr-decay-samples`, not both. |
| --lr-decay-samples | Number of samples for learning rate decay. | int. Choose this parameter or `--lr-decay-iters`, not both. |
| --lr-warmup-fraction | Proportion of learning rate warmup in the total training. | float (0 to 1). Mutually exclusive with `--lr-warmup-iters` and `--lr-warmup-samples`. |
| --lr-warmup-init | Initial value of learning rate warmup. | float, default 0.0. |
| --lr-warmup-iters | Number of iterations for learning rate warmup. | int. Mutually exclusive with `--lr-warmup-fraction`. |
| --lr-warmup-samples | Number of samples for learning rate warmup. | int. Mutually exclusive with `--lr-warmup-fraction`. |
| --lr-wsd-decay-style | Sub-strategy of the WSD decay phase. | Options: `exponential`, `linear`, `cosine`, `minus_sqrt`. Default: `exponential`. Requires `--lr-decay-style WSD`. |
| --lr-wsd-decay-iters | Number of iterations in the WSD decay phase. | int. Requires `--lr-decay-style WSD`. |
| --lr-wsd-decay-samples | Number of samples in the WSD decay phase. | int. Requires `--lr-decay-style WSD`. |
| --weight-decay | Weight decay coefficient. | float, default 0.01. |
| --weight-decay-incr-style | Weight decay increase strategy. | Options: `constant`, `linear`, `cosine`. Default: `constant`. |
| --start-weight-decay | Start value of weight decay. | float. Requires `--weight-decay-incr-style` to be a value other than `constant`. |
| --end-weight-decay | End value of weight decay. | float. Requires `--weight-decay-incr-style` to be a value other than `constant`. |
| --no-weight-decay-cond-type | Type of the no-weight-decay condition. | str. |

## Recompute Strategies

| Parameter Name | Description | Constraint |
|--------|------|------|
| --recompute-activations | Recompute activation switch. | Boolean switch. Equivalent to `--recompute-granularity selective`. |
| --recompute-granularity | Recompute granularity. | Options: `full` (recomputes entire layers), `selective` (recomputes only part of the data). When it is configured together with `--recompute-activations`, selective recomputation takes effect. |
| --recompute-method | Recompute method. | Options: `uniform` (uniform layer partitioning), `block` (by block). Available only with `--recompute-granularity full`. You must also configure `--recompute-num-layers`. |
| --recompute-num-layers | Number of layers contained in each recompute block. | int. Requires `--recompute-granularity full` and `--recompute-method`. |
| --recompute-modules | Modules that need recomputation. | Options: `core_attn`, `moe_act`, `layernorm`, `mla_up_proj`, `mlp`, `moe`, `shared_experts`. Multiple values are allowed. |
| --distribute-saved-activations | Distribute saved activations across multiple devices (requires TP > 1). | Boolean switch. Requires `--recompute-granularity full` and `--recompute-method`. Mutually exclusive with `--sequence-parallel`. |

## Mixture of Experts (MoE)

| Parameter Name | Description | Constraint |
|--------|------|------|
| --num-experts | Number of MoE experts. | int. Required to enable MoE. |
| --moe-layer-freq | Alternation frequency between MoE layers and dense layers. | int, default 1. |
| --moe-ffn-hidden-size | FFN hidden dimension of MoE experts. | int. |
| --moe-latent-size | MoE latent projection dimension. | int. |
| --moe-shared-expert-intermediate-size | Total FFN hidden size of shared experts. | int. |
| --moe-shared-expert-gate | Enable the gating mechanism of shared experts. | Boolean switch. |
| --moe-shared-expert-overlap | Enable overlap between shared expert computation and scheduler communication. | Boolean switch. Requires `--moe-shared-expert-intermediate-size`. Mutually exclusive with `--overlap-moe-expert-parallel-comm`. |
| --moe-router-topk | Number of experts to which each token is routed. | int, default 2. |
| --moe-router-score-function | MoE routing score function. | Options: `softmax`, `sigmoid`, `sqrtsoftplus`. Default: `softmax`. |
| --moe-router-pre-softmax | Enable pre-softmax routing. | Boolean switch. |
| --moe-router-topk-scaling-factor | Routing Top-K scaling factor. | float. Valid only when `--moe-router-pre-softmax` is enabled. |
| --moe-router-num-groups | Number of routing groups. | int, default None. |
| --moe-router-group-topk | Top-K within a routing group. | int. Requires `--moe-router-num-groups`. |
| --moe-router-dtype | Data type for routing computation. | Options: `fp32`, `fp64`. Default: None. |
| --moe-router-load-balancing-type | Routing load balancing strategy. | Options: `aux_loss`, `seq_aux_loss`, `global_aux_loss`, `sinkhorn`, `none`. Default: `aux_loss`. |
| --moe-router-enable-expert-bias | Enable TopK routing with dynamic expert bias. | Boolean switch. |
| --moe-router-bias-update-rate | Expert bias update rate. | float, default 1e-3. Requires `--moe-router-enable-expert-bias`. |
| --moe-router-fusion | Enable fusion of MoE TopK routing and auxiliary loss computation. | Boolean switch. |
| --moe-router-force-load-balancing | [Experimental] Force load balancing routing (for benchmarking only). | Boolean switch. |
| --moe-enable-routing-replay | Enable the routing replay feature. | Boolean switch. |
| --moe-aux-loss-coeff | Auxiliary loss scaling coefficient. | float, default 0.0. |
| --moe-z-loss-coeff | Z-loss scaling coefficient. | float, default None (a suggested starting value is 1e-3). |
| --moe-input-jitter-eps | Input jitter epsilon (ST-MoE input jitter regularization). | float, default None. |
| --moe-token-dispatcher-type | Token dispatcher type. | Options: `allgather`, `alltoall`, `flex`. Default: `allgather`. |
| --moe-permute-fusion | Fuse token permutation operations during token dispatch. | Boolean switch. |
| --moe-expert-capacity-factor | Capacity factor of each expert. | float, default None (no limit). |
| --moe-pad-expert-input-to-capacity | Pad the expert input to the capacity length. | Boolean switch. Requires `--moe-expert-capacity-factor`. |
| --moe-token-drop-policy | Token drop policy. | Options: `probs`, `position`. Requires `--moe-expert-capacity-factor`. |
| --moe-grouped-gemm | Enable grouped GEMM. | Boolean switch. Requires CUTLASS 2.8 or later. |
| --moe-apply-probs-on-input | Apply routing probabilities to the expert input. | Boolean switch. |
| --moe-use-upcycling | Load a dense model checkpoint and convert it to a MoE model. | Boolean switch. Use it with `--moe-upcycling-granularity`. |
| --moe-upcycling-granularity | Reduction factor of the expert hidden size relative to the dense FFN. | int, default 1. Requires `--moe-use-upcycling`. |
| --moe-per-layer-logging | Enable per-layer logging for MoE. | Boolean switch. |

## Checkpointing

### Saving and Loading

| Parameter Name | Description | Constraint |
|--------|------|------|
| --save | Path for saving checkpoints. | str. |
| --save-interval | Checkpoint save interval (save every N steps). | int. |
| --load | Path for loading checkpoints. | str. |
| --no-load-optim | Do not load the optimizer state. | Boolean switch. |
| --no-load-rng | Do not load the random number generator state. | Boolean switch. |

### Checkpoint Format and Conversion

| Parameter Name | Description | Constraint |
|--------|------|------|
| --ckpt-format | Format for saving checkpoints. | Options: `torch`, `torch_dist`, `torch_dcp`, `fsdp_dtensor`. Default: `torch_dist`. |
| --ckpt-convert-format | Target format for checkpoint conversion. | str. |
| --ckpt-convert-save | Convert to the specified format on save. | str. |
| --ckpt-convert-update-legacy-dist-opt-format | Update the legacy distributed optimizer format during conversion. | Boolean switch. |
| --auto-detect-ckpt-format | Automatically detect the checkpoint format. | Boolean switch. |
| --dist-ckpt-optim-fully-reshardable | Save the optimizer state as fully reshardable. | Boolean switch. |
| --dist-ckpt-strictness | Loading strictness for distributed checkpoints. | Options: `assume_ok_unexpected`, `log_unexpected`, `log_all`, `raise_unexpected`, `raise_all`, `return_unexpected`, `return_all`, `ignore_all`. Default: `assume_ok_unexpected`. |

### Parallel Saving and Loading

| Parameter Name | Description | Constraint |
|--------|------|------|
| --async-save | Enable asynchronous save (does not block training). | Boolean switch. Requires `--use-persistent-ckpt-worker`. Supports only the `torch_dist` format. |
| --ckpt-fully-parallel-save | Enable fully parallel save. | Boolean switch (deprecated). Mutually exclusive with `--no-ckpt-fully-parallel-save`. |
| --no-ckpt-fully-parallel-save | Disable fully parallel save. | Boolean switch. Mutually exclusive with `--ckpt-fully-parallel-save`. |
| --ckpt-fully-parallel-load | Load checkpoints in a fully parallel manner. | Boolean switch. |
| --ckpt-fully-parallel-load-exchange-algo | Exchange algorithm for fully parallel load. | Options: `broadcast`, `gather_rounds`, `gather_object`. Default: `broadcast`. |
| --ckpt-fully-parallel-save-process-group | Process group for fully parallel save. | str. |
| --ckpt-fully-parallel-load-process-group | Process group for fully parallel load. | str. |
| --ckpt-assume-constant-structure | Assume that the checkpoint model structure is constant and skip structure checks. | Boolean switch. |
| --use-persistent-ckpt-worker | Use a persistent checkpoint worker thread. | Boolean switch. |

### Fine-Tuning and Pre-Training

| Parameter Name | Description | Constraint |
|--------|------|------|
| --finetune | Enable fine-tuning mode (does not load the optimizer or scheduler state). | Boolean switch. |
| --pretrained-checkpoint | Path of the pre-trained checkpoint. | str. |
| --ckpt-step | Specific step of the checkpoint to load. | int. |
| --use-checkpoint-args | Use the training parameters saved in the checkpoint. | Boolean switch. |
| --use-mp-args-from-checkpoint-args | Use the model parallel parameters saved in the checkpoint. | Boolean switch. Requires `--use-checkpoint-args`. |
| --exit-on-missing-checkpoint | Exit the program when the checkpoint does not exist. | Boolean switch. |

## CPU Offload

| Parameter Name | Description | Constraint |
|--------|------|------|
| --cpu-offloading-num-layers | Number of layers that need to be offloaded to the CPU. | int, default 0. |
| --optimizer-cpu-offload | Enable CPU offload of the optimizer state. | Boolean switch. |
| --optimizer-offload-fraction | Proportion of the optimizer offloaded to the CPU. | float (0 to 1), default 1.0. Requires `--optimizer-cpu-offload`. |
| --use-torch-optimizer-for-cpu-offload | Use the PyTorch optimizer for CPU offload. | Boolean switch. Requires `--optimizer-cpu-offload`. |
| --overlap-cpu-optimizer-d2h-h2d | Overlap D2H/H2D data transfer of the CPU optimizer. | Boolean switch. Requires `--optimizer-cpu-offload`. |
| --no-pin-cpu-grads | Do not pin CPU gradient memory (disables pinned memory). | Boolean switch. |
| --no-pin-cpu-params | Do not pin CPU parameter memory (disables pinned memory). | Boolean switch. |

## Tokenizer and Datasets

### Tokenizer

| Parameter Name | Description | Constraint |
|--------|------|------|
| --tokenizer-type | Tokenizer type. | Options: many (for example, `BertWordPieceLowerCase`, `GPT2BPETokenizer`, and `HuggingFaceTokenizer`). For the complete list, see `--help` of the corresponding version. |
| --tokenizer-model | Path of the tokenizer model. | str. |
| --tokenizer-metadata | Path of the tokenizer metadata JSON file. | str. |
| --tokenizer-special-tokens | List of special tokens. | str. |
| --tiktoken-pattern | Tiktoken pattern. | Options: `v1`, `v2`. |
| --tiktoken-num-special-tokens | Number of Tiktoken special tokens. | int. |
| --tokenizer-sentencepiece-legacy | Use the legacy SentencePiece compatibility mode. | Boolean switch. |
| --trust-remote-code | Trust remote code (required when you load a custom tokenizer). | Boolean switch. |
| --null-tokenizer-eod-id | Null Tokenizer EOD ID. | int. |
| --null-tokenizer-pad-id | Null Tokenizer Pad ID. | int. |
| --vocab-size | Vocabulary size. | int. |
| --padded-vocab-size | Padded vocabulary size. | int. |
| --vocab-file | Path of the vocabulary file. | str. |
| --merge-file | Path of the BPE merge file. | str. |
| --vocab-extra-ids | Number of extra vocabulary IDs. | int. |
| --make-vocab-size-divisible-by | Alignment value of the vocabulary size. | int. |

### Datasets and Data Loading

| Parameter Name | Description | Constraint |
|--------|------|------|
| --data-path | Data path. | str. Mutually exclusive with `--data-args-path` and `--mock-data`. |
| --train-data-path | Training data path. | str. |
| --valid-data-path | Validation data path. | str. |
| --test-data-path | Test data path. | str. |
| --data-args-path | Path of the data argument file (replaces `--data-path`). | str. Mutually exclusive with `--data-path`. |
| --per-split-data-args-path | Path of the JSON file for the data arguments of each split. | str. |
| --per-dataset-sequences-path | Path of the JSON file for the sequences of each dataset. | str. |
| --split | Proportions of the training, validation, and test datasets. | str, for example `"99,1,0"`. |
| --seq-length | Sequence length. | int. |
| --encoder-seq-length | Encoder sequence length. | int. Mutually exclusive with `--seq-length`. |
| --decoder-seq-length | Decoder sequence length. | int. |
| --dataloader-type | Data loader type. | str. |
| --dataloader-fast-cache-load | Enable the fast cache load path. | Boolean switch. Requires `--data-cache-path`. Mutually exclusive with `--data-path`. |
| --dataloader-defer-npy-index-mmap | Defer the memory mapping of .npy files. | Boolean switch. |
| --data-cache-path | Path of the data cache index file. | str. |
| --no-mmap-bin-files | Do not memory-map binary files. | Boolean switch. |
| --mock-data | Use mock data (skips data loading and validation). | Boolean switch. Mutually exclusive with `--data-path` and similar parameters. Do not use it in production. |
| --num-workers | Number of data loading worker threads. | int. |
| --num-dataset-builder-threads | Number of dataset builder threads. | int. |
| --object-storage-cache-path | Object storage cache path. | str. |
| --mid-level-dataset-surplus | Redundancy ratio of mid-level datasets. | float. |
| --sample-rate | Sampling rate. | float. |
| --mask-prob | Mask probability. | float. |
| --short-seq-prob | Short sequence probability. | float. |
| --reset-position-ids | Reset position IDs after the end-of-document token. | Boolean switch. |
| --reset-attention-mask | Reset the attention mask after the end-of-document token. | Boolean switch. |
| --eod-mask-loss | Mask the loss at the end-of-document positions. | Boolean switch. |
| --no-create-attention-mask-in-dataloader | Do not create the attention mask in the data loader. | Boolean switch. |
| --allow-ambiguous-pad-tokens | Allow ambiguous padding tokens. | Boolean switch. |
| --phase-transition-iterations | Phase transition iterations. | str. |

### Fill-in-the-Middle (FIM)

| Parameter Name | Description | Constraint |
|--------|------|------|
| --fim-data | Enable FIM data. | Boolean switch. |
| --fim-rate | FIM conversion probability. | float. Requires `--fim-data`. |
| --fim-spm-rate | Probability of the FIM SPM format. | float. Requires `--fim-data`. |
| --fim-split-sample | FIM sample split string. | str. Requires `--fim-data`. |
| --fim-fragment-rate | FIM fragment proportion. | float. Requires `--fim-data`. |
| --fim-no-prefix | FIM no-prefix marker. | Boolean switch. Requires `--fim-data`. |
| --fim-prefix-token | FIM prefix token. | int. Requires `--fim-data`. |
| --fim-middle-token | FIM middle token. | int. Requires `--fim-data`. |
| --fim-suffix-token | FIM suffix token. | int. Requires `--fim-data`. |
| --fim-pad-token | FIM padding token. | int. Requires `--fim-data`. |
| --fim-eod-token | FIM end-of-document token. | int. Requires `--fim-data`. |

## Hybrid Models (Mamba + Transformer)

| Parameter Name | Description | Constraint |
|--------|------|------|
| --mamba-state-dim | Mamba state dimension. | int, default 128. Required when Mamba layers are enabled. |
| --mamba-head-dim | Mamba head dimension. | int, default 64. Required when Mamba layers are enabled. |
| --mamba-num-groups | Number of Mamba groups. | int, default 8. Required when Mamba layers are enabled. |
| --mamba-num-heads | Number of Mamba heads. | int. |
| --mtp-num-layers | Number of multi-token prediction (MTP) layers. | int, default None (disabled). |
| --mtp-loss-scaling-factor | MTP loss scaling factor. | float, default 0.1. Requires `--mtp-num-layers > 0`. |
| --mtp-standalone | MTP standalone mode. | Boolean switch. Requires `--mtp-num-layers > 0`. |
| --calculate-per-token-loss | Compute the loss for each token instead of averaging over the sequence. | Boolean switch. |

## Distributed Communication Optimization

| Parameter Name | Description | Constraint |
|--------|------|------|
| --use-tp-pp-dp-mapping | Use the TP/PP/DP mapping. | Boolean switch. |
| --overlap-grad-reduce | Overlap gradient reduction with computation. | Boolean switch. |
| --overlap-param-gather | Overlap parameter gather with computation. | Boolean switch. Requires the distributed optimizer or FSDP. You must also enable `--overlap-grad-reduce`. |
| --overlap-param-gather-with-optimizer-step | Overlap parameter gather with the optimizer step. | Boolean switch. Requires `--overlap-param-gather`. |
| --overlap-dispatch-backward-with-experts-wgrad | Overlap backward dispatch with expert wgrad. | Boolean switch. |
| --distributed-backend | Distributed communication backend. | Options: `nccl`, `gloo`. Default: `nccl`. |
| --distributed-timeout-minutes | Distributed communication timeout. | int, default 10 (minutes). |
| --use-sharp | Enable SHARP communication acceleration. | Boolean switch. |
| --sharp-enabled-group | SHARP enabled group. | Options: `dp`, `dp_replica`. Requires `--use-sharp`. |
| --create-all-gather-group | Create the all-gather group. | Boolean switch. |
| --suggested-communication-unit-size | Suggested communication unit size. | int. |
| --disable-symmetric-registration | Disable symmetric registration. | Boolean switch. |
| --symmetric-ar-type | All-reduce type for symmetric memory. | str. |
| --local-rank | Local rank. | int. |
| --no-align-grad-reduce | Disable gradient reduction alignment. | Boolean switch. |
| --no-align-param-gather | Disable parameter gather alignment. | Boolean switch. |
| --wgrad-deferral-limit | wgrad deferral limit. | int, default 0. |
| --lazy-mpu-init | Defer MPU initialization. | Boolean switch. |
| --disable-gloo-process-groups | Disable Gloo process groups. | Boolean switch. |

## Fusion Optimization

| Parameter Name | Description | Constraint |
|--------|------|------|
| --no-gradient-reduce-div-fusion | Disable gradient reduce division fusion. | Boolean switch. |
| --no-gradient-accumulation-fusion | Disable gradient accumulation fusion. | Boolean switch. |
| --cross-entropy-loss-fusion | Enable cross entropy loss fusion. | Boolean switch. |
| --cross-entropy-fusion-impl | Implementation of cross entropy fusion. | Options: `native`, `te`. Requires `--cross-entropy-loss-fusion`. |
| --no-masked-softmax-fusion | Disable masked softmax fusion. | Boolean switch. |
| --no-bias-dropout-fusion | Disable bias-dropout fusion. | Boolean switch. |
| --fused-residual-rmsnorm | Use the fused RMSNorm plus residual connection kernel. | Boolean switch. Requires `--normalization RMSNorm`. |
| --disable-jit-fuser | Disable the JIT fuser. | Boolean switch. |

## Training Control

### Batches and Iterations

| Parameter Name | Description | Constraint |
|--------|------|------|
| --micro-batch-size | Micro-batch size (number of samples processed by each GPU per step). | int, required. |
| --global-batch-size | Global batch size. | int, required. It must be a multiple of micro-batch-size × DP. |
| --rampup-batch-size | Batch rampup configuration. | str. Mutually exclusive with `--train-iters`. |
| --step-batch-size-schedule | MBS schedule for each step. | str. |
| --decrease-batch-size-if-needed | Automatically decrease the batch size on OOM. | Boolean switch. |
| --train-iters | Number of training iterations. | int. Mutually exclusive with `--train-samples`. |
| --train-samples | Number of training samples. | int. Mutually exclusive with `--train-iters`. |
| --eval-iters | Number of evaluation iterations. | int. |
| --eval-interval | Evaluation interval (evaluate every N steps). | int, default None. |
| --start-eval-at-iter | Iteration at which evaluation starts. | int. |
| --eval-global-batch-size | Evaluation global batch size. | int. |
| --eval-micro-batch-size | Evaluation micro-batch size. | int. |
| --iterations-to-skip | Number of initial iterations to skip. | int. |
| --skip-train | Skip training (evaluation only). | Boolean switch. |
| --full-validation | Enable full validation. | Boolean switch. |
| --multiple-validation-sets | Enable multiple validation sets. | Boolean switch. |

### Model Architecture

| Parameter Name | Description | Constraint |
|--------|------|------|
| --num-layers | Number of Transformer layers. | int, required. |
| --encoder-num-layers | Number of encoder layers. | int. |
| --decoder-num-layers | Number of decoder layers. | int. |
| --hidden-size | Hidden layer dimension. | int, required. |
| --ffn-hidden-size | FFN hidden layer dimension. | int. Defaults to `hidden_size` × 4. |
| --num-attention-heads | Number of attention heads. | int, required. |
| --num-query-groups | Number of query groups (used with GQA). | int. |
| --kv-channels | Number of KV channels. | int. Defaults to `hidden_size` / `num_attention_heads`. |
| --untie-embeddings-and-output-weights | Untie embeddings and output weights. | Boolean switch. |
| --attention-dropout | Attention dropout probability. | float, default 0.1. |
| --hidden-dropout | Hidden layer dropout probability. | float, default 0.1. |
| --use-mcore-models | Use the Megatron-Core model implementation. | Boolean switch (deprecated, mcore is now the default). |
| --use-legacy-models | Use the legacy model implementation. | Removed from the code. |
| --transformer-impl | Transformer implementation. | Options: `local`, `transformer_engine`, `inference_optimized`. Default: `transformer_engine`. |
| --heterogeneous-layers-config-path | Path of the heterogeneous layer configuration. | str. |
| --heterogeneous-layers-config-encoded-json | Heterogeneous layer configuration JSON. | str. Mutually exclusive with `--heterogeneous-layers-config-path`. |

### Initialization

| Parameter Name | Description | Constraint |
|--------|------|------|
| --seed | Random seed. | int, default 1234. |
| --init-method-std | Initialization standard deviation. | float, default 0.02. |
| --embedding-init-method-std | Embedding initialization standard deviation. | float. |
| --init-model-with-meta-device | Initialize the model with the meta device. | Boolean switch. |
| --no-initialization | Skip initialization. | Boolean switch. |
| --use-cpu-initialization | Complete model initialization on the CPU. | Boolean switch. |

### Gradient Control

| Parameter Name | Description | Constraint |
|--------|------|------|
| --clip-grad | Gradient clipping threshold (clip by global norm). | float, default 1.0. |
| --no-check-for-nan-in-loss-and-grad | Do not check for NaN in the loss and gradients. | Boolean switch. |
| --check-for-large-grads | Check for large gradients. | Boolean switch. |
| --check-for-spiky-loss | Detect loss spikes. | Boolean switch. |

### Memory Management

| Parameter Name | Description | Constraint |
|--------|------|------|
| --manual-gc | Enable manual garbage collection. | Boolean switch. |
| --manual-gc-interval | GC interval (run once every N steps). | int, default 0. Requires `--manual-gc`. |
| --no-manual-gc-eval | Do not run GC during evaluation. | Boolean switch. Requires `--manual-gc`. |
| --empty-unused-memory-level | Memory cleanup level. | Options: 0, 1, 2. Default: 0. |
| --memory-snapshot-path | Memory snapshot path. | str, default `snapshot.pickle`. |

### Training Stability and Debugging

| Parameter Name | Description | Constraint |
|--------|------|------|
| --deterministic-mode | Enable deterministic mode (trades performance for reproducibility). | Boolean switch. |
| --npu-deterministic | Enable full-chain deterministic computation on NPU, which covers operator-level determinism of HCCL communication, disabling Matmul K-axis translation, enabling PyTorch deterministic algorithms, and fixing the `torch_npu` RNG seed (automatically injected after Megatron seed initialization). It also writes an audit value for PYTHONHASHSEED. | Boolean switch. Effective only with `--optimization-level >= 2`. Must be used with a fixed `--seed`. If you need to fix Python hash randomization, you must run `export PYTHONHASHSEED=<seed>` before you start the training process (writing it at runtime does not take effect). This parameter trades performance for reproducibility. Therefore, you are advised to enable it only during troubleshooting or precision comparison. It is complementary to `--deterministic-mode` and both can be enabled at the same time. |
| --rerun-mode | Rerun policy when a step anomaly is detected. | Options: `disabled`, `validate_results`, `report_stats`. Default: `validate_results`. |
| --error-injection-rate | Random fault injection rate. | int, default 0. |
| --error-injection-type | Random fault injection type. | Options: `correct_result`, `transient_error`, `persistent_error`. Default: `transient_error`. |
| --straggler-ctrlr-port | Communication port for straggler detection. | int. |
| --straggler-minmax-count | Number of samples retained for straggler statistics. | int. |
| --disable-straggler-on-startup | Disable straggler detection at startup. | Boolean switch. |
| --check-weight-hash-across-dp-replicas-interval | DP weight verification interval. | int. |
| --train-sync-interval | Training synchronization interval. | int. |
| --result-rejected-tracker-filename | Result rejection tracker file. | str. |
| --batch-invariant-mode | Enable batch invariant mode. | Boolean switch. |
| --barrier-with-L1-time | Enable the level-1 timing barrier. | Boolean switch. |
| --no-barrier-with-level-1-timing | Disable the level-1 timing barrier. | Boolean switch. Mutually exclusive with `--barrier-with-L1-time`. |
| --test-mode | Enable test mode. | Boolean switch. |
| --bert-no-binary-head | BERT without a binary head. | Boolean switch. |
| --dump-param-to-param-group-map | Dump path for the parameter-to-parameter-group map. | str. |

### Exit Control

| Parameter Name | Description | Constraint |
|--------|------|------|
| --exit-interval | Exit interval (check whether to exit every N steps). | int. |
| --exit-duration-in-mins | Exit duration (exit after training runs for more than N minutes). | int (minutes). |
| --exit-signal-handler | Enable the exit signal handler. | Boolean switch. |
| --exit-signal-handler-for-dataloader | Enable the exit signal handler for the data loader. | Boolean switch. |

### Other Training Parameters

| Parameter Name | Description | Constraint |
|--------|------|------|
| --mlp-chunks-for-prefill | Number of MLP chunks in the prefill phase. | int. |

## Logging and Output

| Parameter Name | Description | Constraint |
|--------|------|------|
| --log-interval | Log printing interval (print every N steps). | int, default 100. |
| --log-params-norm | Log parameter norms. | Boolean switch. |
| --log-throughput | Log performance metrics such as throughput and TFLOPS. | Boolean switch. |
| --log-progress | Log training progress. | Boolean switch. |
| --log-num-zeros-in-grad | Log the number of zeros in the gradients. | Boolean switch. |
| --logging-level | Log output level. | str. |
| --timing-log-level | Timing log level. | Options: 0 (off), 1 (basic), 2 (detailed). |
| --timing-log-option | Timing log aggregation method. | Options: `max`, `minmax`, `all`. Default: `minmax`. |
| --tensorboard-dir | TensorBoard log output directory. | str. |
| --tensorboard-log-interval | TensorBoard logging interval. | int, default 1. |
| --tensorboard-queue-size | Queue size for TensorBoard event writes. | int, default 1000. |
| --log-timers-to-tensorboard | Log the duration of each phase to TensorBoard. | Boolean switch. Requires `--tensorboard-dir`. |
| --no-log-loss-scale-to-tensorboard | Do not log the loss scale factor to TensorBoard. | Boolean switch. |
| --log-validation-ppl-to-tensorboard | Write validation PPL to TensorBoard. | Boolean switch. Requires `--tensorboard-dir`. |
| --log-memory-to-tensorboard | Log host memory to TensorBoard. | Boolean switch. Requires `--tensorboard-dir`. |
| --log-memory-interval | Host memory logging interval. | int, default None. Requires `--log-memory-to-tensorboard`. |
| --log-world-size-to-tensorboard | Log the number of ranks to TensorBoard. | Boolean switch. Requires `--tensorboard-dir`. |

### Performance Profiling (NPU Profiling)

The native Megatron `--profile` parameter is based on `torch.profiler` (CUDA/CPU) and cannot collect device-side data on Ascend NPU. The `profile` feature of MegatronAdaptor (effective when `optimization_level>=2` and `--profile` is passed) automatically redirects it to `torch_npu.profiler` collection. Traces are written to drive in the TensorBoard format, and you can view them with MindStudio or TensorBoard.

| Parameter Name | Description | Constraint |
|--------|------|------|
| --profile | Enable performance profiling (automatically redirected to `torch_npu.profiler` collection on NPU). | Boolean switch, default False. Effective only with `--optimization-level >= 2`. It has zero impact on training when it is not enabled. |
| --profile-ranks | Set of ranks that collect profiling data. `-1` means all ranks. | list of int, default [0] (Megatron native parameter). |
| --profile-step-start | Start iteration of the collection window. | int, default 10 (Megatron native parameter). |
| --profile-step-end | End iteration of the collection window (exclusive). | int, default 12 (Megatron native parameter). It must be greater than `--profile-step-start`. |
| --profile-level | Profiler collection depth. | Options: `level0`, `level1`, `level2`. Default: `level0`. A higher level means finer collection granularity and higher memory overhead. |
| --profile-with-cpu | Collect CPU activity at the same time, which aligns host-device timelines. | Boolean switch, default False. |
| --profile-with-stack | Record the Python call stack. | Boolean switch, default False. |
| --profile-with-memory | Record memory allocation events. | Boolean switch, default False. |
| --profile-record-shapes | Record the input shapes of operators. | Boolean switch, default False. |
| --profile-save-path | Directory where traces are written. | str, default `./profile_dir`. |

### Precision Comparison (NPU DataDump)

When you migrate across platforms (GPU to NPU), whole-network precision comparison requires layer-by-layer comparison of forward and backward values. The `npu_datadump` feature of MegatronAdaptor (effective when `optimization_level>=2` and `--npu-datadump` is passed) integrates the `PrecisionDebugger` of the MindStudio Training Tools (MSTT) msprobe tool. It starts collection and injects into the model before `train_step`, and finalizes collection and advances the step count after the end-of-step hook of each training step, which enables one-click whole-network precision data dump. Install msprobe before use: `pip install mindstudio-probe`. The Lite backend is not supported yet.

| Parameter Name | Description | Constraint |
|--------|------|------|
| --npu-datadump | Enable whole-network precision data dump (msprobe `PrecisionDebugger` collection). | Boolean switch, default False. Effective only with `--optimization-level >= 2`. When it is not enabled, msprobe is not imported and training is not affected. |

The collection scope (rank, step, collection mode, and so on) is adjusted through the `config.json` file shipped with the package (for source installation: `megatron_adaptor/patches/megatron/npu_datadump/config.json`, for pip installation: the corresponding directory in site-packages). By default, it collects statistics precision data (`task=statistics`, `level=L1`) of RANK0 and STEP0 and writes the data to `./dump_path` in the process working directory. After the dump is complete, you can use the `compare.json` template shipped with the package (configure `npu_path` and `bench_path`) to compare precision through msprobe. For configuration details, see the [official msprobe documentation](https://gitcode.com/Ascend/msprobe).

> [!NOTE]
>
> - msprobe does not support the dynamo scenario in PyTorch 2.7 and later versions yet.
> - For single-device training, you must change `rank` in `config.json` to an empty list `[]` (specifying a rank results in empty dump data).
> - In precision comparison scenarios, you can use the Megatron `--deterministic-mode` parameter to fix computation randomness.

## Maximal Update Parametrization (µP)

| Parameter Name | Description | Constraint |
|--------|------|------|
| --use-mup | Enable µP parametrization. | Boolean switch. |
| --mup-width-mult | Width multiplier. | float, default 1.0 (automatically computed as `hidden_size` / `mup_base_hidden_size`). Requires `--use-mup`. |
| --mup-base-hidden-size | Base hidden layer size. | int. Requires `--use-mup`. |
| --mup-embedding-mult | Embedding layer multiplier. | float, default 1.0. Requires `--use-mup`. |
| --mup-output-mult | Output layer multiplier. | float, default 1.0 (automatically set to 1/width_mult when MuP is enabled). Requires `--use-mup`. |
| --mup-base-head-dim | Base attention head dimension. | int. Requires `--use-mup`. |
| --mup-attn-scale-power | Attention scaling power. | float, default 1.0 (0.5 = standard attention, 1.0 = MuP). Requires `--use-mup`. |

## Summary

MegatronAdaptor is currently verified to cover the feature categories listed above, which include the core capabilities of Megatron-LM:

- **Core training**: activation functions, attention mechanisms, position embeddings, normalization
- **Precision optimization**: BF16/FP16/FP8 mixed precision, dynamic loss scale adjustment
- **Parallelism strategies**: TP/PP/SP/CP/EP multi-dimensional parallelism, Virtual Pipeline, FSDP
- **Optimizers**: Adam/SGD/Muon/Lion/SOAP, distributed optimizer, CPU Offload
- **Memory optimization**: recomputation, checkpointing, layer offload
- **Model architecture**: MoE, Hybrid (Mamba+Transformer), MTP, µP
- **Data processing**: multiple tokenizers, data cache, FIM training
- **Communication optimization**: TP Comm Overlap, SHARP, communication overlap
- **Training stability**: straggler detection, fault injection, deterministic mode, gradient check
- **Observability**: TensorBoard integration, throughput/memory/timing logs, NPU Profiling (in-process `torch_npu.profiler` collection), precision comparison (whole-network precision data dump with the msprobe `PrecisionDebugger`)
