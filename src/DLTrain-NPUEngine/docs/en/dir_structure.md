# TransformerEngineNPU Project Directory Structure

```plaintext
TransformerEngineNPU/
├── README.md                            # Project documentation, introducing the purpose, version compatibility, and installation method of TransformerEngineNPU
├── LICENSE                              # License file
├── setup.py                             # Installation and packaging script, defining package dependencies and the build entry point
├── pyproject.toml                       # Build system configuration, declaring the build backend and build-time dependencies
├── docs/                                # Project documentation directory
│   ├── zh/                              # Chinese documentation, including introduction, software installation, API list, FAQs, and security statement
│   └── en/                              # English documentation directory, maintained by the translation team
├── transformer_engine/                  # Main Python package directory, containing all core implementation code
│   ├── common/                          # Framework-agnostic common modules
│   │   └── recipe/                      # Quantization strategies, defining scaling strategies such as DelayedScaling, MXFP8, and BlockScaling
│   ├── debug/                           # Numerical debugging tools, supporting online observation and comparison of quantization behavior
│   │   └── pytorch/                     # Debugging implementation for the PyTorch backend
│   │       ├── debug_state.py           # Debug state, maintaining the global state of numerical debugging
│   │       ├── debug_quantization.py    # Debug quantizer, containing the numerical observation logic of the quantization process
│   │       └── utils.py                 # Debug utility functions
│   └── pytorch/                         # PyTorch backend, the main implementation of TENPU's external interfaces
│       ├── constants.py                 # Constant definitions, containing global switches such as the attention backend
│       ├── cpu_offload.py               # CPU Offload context, providing activation/weight offload capabilities
│       ├── distributed.py               # Distributed utilities, providing functions such as checkpoint and TP attribute settings
│       ├── fp8.py                       # FP8 metadata, defining the infrastructure related to quantization formats
│       ├── float8_tensor.py             # Float8 tensor (legacy path), implementation retained for compatibility
│       ├── jit.py                       # JIT utilities, supporting compilation paths such as torch.compile
│       ├── permutation.py               # MoE permutation, providing token reordering and restoration primitives
│       ├── quantized_tensor.py          # Quantized tensor base class, defining the QuantizedTensor and Quantizer abstractions
│       ├── router.py                    # MoE routing, providing fused top-k and auxiliary loss computation
│       ├── torch_version.py             # Version adaptation, handling compatibility differences across torch versions
│       ├── utils.py                     # General utility functions
│       ├── attention/                   # Attention module
│       │   ├── rope.py                  # RoPE rotary position encoding, including fused implementations
│       │   └── dot_product_attention/   # Fused attention implementation, including context parallelism strategies such as Ulysses/Ring/KV-AllGather/Swap
│       ├── module/                      # Foundational layer modules
│       │   ├── base.py                  # Module base class, providing FP8 state management and UserBuffer initialization
│       │   ├── linear.py                # Linear fully connected layer, supporting FP8/BF16 and delayed wgrad
│       │   ├── grouped_linear.py        # GroupedLinear grouped fully connected layer, serving MoE scenarios
│       │   ├── layernorm.py             # LayerNorm normalization layer
│       │   ├── rmsnorm.py               # RMSNorm normalization layer
│       │   ├── layernorm_linear.py      # LayerNormLinear, fusing normalization and linear layers
│       │   ├── layernorm_mlp.py         # LayerNormMLP, fusing normalization and a two-layer MLP
│       │   ├── fp8_padding.py           # Fp8Padding, token-dimension padding friendly to FP8 GEMM
│       │   └── fp8_unpadding.py         # Fp8Unpadding, the corresponding unpadding paired with Fp8Padding
│       ├── ops/                         # Low-level operator wrappers
│       │   ├── basic/                   # Basic operators, per-operator reference implementations
│       │   ├── fused/                   # Fused operators, cross-operator fused implementations
│       │   ├── fuser.py                 # Operator fuser, selects the fusion path based on configuration
│       │   └── gemm_comparison.py       # GEMM numerical comparison, online verification of quantized GEMM against the reference implementation
│       ├── optimizers/                  # Optimizer operators
│       │   └── multi_tensor_apply.py    # Multi-tensor fused operator, merges similar operations in optimizer steps
│       ├── quantization/                # Quantization management
│       │   ├── manager.py               # Quantization manager, provides contexts such as autocast and fp8_model_init
│       │   ├── utils.py                 # Quantization capability query, such as is_fp8_available
│       │   ├── quantizer_role.py        # Quantizer role definition, supporting CustomRecipe
│       │   └── state/                   # Policy runtime state, RecipeState implementation for each recipe
│       ├── tensor/                      # Quantized tensors
│       │   ├── float8_tensor.py         # Float8Tensor and Float8Quantizer implementation
│       │   ├── float8_blockwise_tensor.py # 128×128 block-wise scaled FP8 tensor
│       │   ├── mxfp8_tensor.py          # MXFP8 scaled tensor
│       │   ├── mxfp4_tensor.py          # MXFP4 scaled tensor (Ascend extension)
│       │   ├── hif8_tensor.py           # HIF8 tensor (Ascend extension)
│       │   ├── w4a8_tensor.py           # W4A8 weight tensor (Ascend extension)
│       │   └── storage/                 # Tensor storage layer, defining serialization and underlying storage of quantized tensors
│       └── triton/                      # triton-ascend implementation, operator kernels based on Triton
├── tests/                               # Test directory
│   ├── pytorch/                         # PyTorch interface unit tests
│   ├── atk/                             # ATK-related tests
│   └── system_tests/                    # System-level tests
├── build_tools/                         # Build tools, maintaining the version number file
├── ci/                                  # Continuous integration scripts
└── pre-commit/                          # Pre-commit check configuration, including ruff and spell check rules
```
