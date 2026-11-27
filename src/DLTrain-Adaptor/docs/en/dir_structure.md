# MegatronAdaptor Project Directory Structure

```plaintext
MegatronAdaptor/
├── README.md                            # Project documentation that describes MegatronAdaptor features, matching versions, and usage
├── setup.py                             # Packaging configuration that defines the installation and build rules for the megatron_adaptor package
├── docs/                                # Project documentation directory that contains the project introduction, installation guide, quick start, supported features, and so on
├── megatron_adaptor/                    # Core source code directory that holds all adaptation implementations
│   ├── __init__.py                      # Package entry point that calls patch_features() on import to apply all adaptation patches automatically
│   ├── megatron_adaptor.py              # Main patch entry point where patch_features() coordinates the two-phase patch application and feature gating
│   ├── auto_execute.py                  # Import-time execution control that controls whether adaptation runs automatically without importing this package
│   ├── features_manager/                # Feature manager that defines parameter registration and patch registration for each feature
│   │   ├── mindspeed_feature.py         # Feature base class that defines common interfaces such as optimization_level gating and patch application status
│   │   ├── features_manager.py          # Feature list management that uniformly applies pre and post patches, validates parameters, and removes patches
│   │   ├── requirements/                # Adaptation features for the basic runtime environment
│   │   ├── megatron/                    # Basic adaptation features for Megatron-LM
│   │   ├── transformer_engine/          # Adaptation features for TransformerEngineNPU
│   │   └── functional/                  # Functional features
│   ├── patches/                         # Patch implementation directory that stores the replacement functions of each patch
│   │   ├── patch_manager.py             # Patch manager that handles PatchesManager.register_patch and replacement of target symbols
│   │   ├── patch_utils.py               # Patch utilities that probe the availability of upstream modules such as megatron.training
│   │   ├── requirements/                # Patch implementations for the basic runtime environment
│   │   ├── megatron/                    # Megatron-LM patch implementations
│   │   └── transformer_engine/          # TransformerEngineNPU patch implementations
│   └── utils/                           # General-purpose utilities directory
│       └── args_utils.py                # Parameter utilities that provide ContextVar access to the full set of parameters and manage the feature parameter cache
├── test/                                # Test directory that contains all unit tests and system tests
├── ci/                                  # Continuous integration directory
```
