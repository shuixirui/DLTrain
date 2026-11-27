"""Upstream baselines used by the integrated training components."""

COMPONENTS = (
    {
        "name": "Megatron-LM",
        "directory": "src/DLTrain-Megatron-LM",
        "commit": "ba7b5ebce12af60627a80985792a1449ce45f46c",
        "entry_point": "megatron/core/__init__.py",
    },
    {
        "name": "MegatronAdaptor",
        "directory": "src/DLTrain-MegatronAdaptor",
        "commit": "59653dff2df7bf03b528a7fb8a7f1fb8a288f741",
        "entry_point": "megatron_adaptor/__init__.py",
    },
    {
        "name": "TransformerEngineNPU",
        "directory": "src/DLTrain-TransformerEngineNPU-2.17",
        "commit": "09f6cc1a17314112bc1cdeef1e061c724f256f38",
        "entry_point": "transformer_engine/pytorch/__init__.py",
    },
    {
        "name": "Megatron-Bridge",
        "directory": "src/DLTrain-Megatron-Bridge-interns2",
        "commit": "a069bdffa41fdc4d7cb377e72a89d484167b8a85",
        "entry_point": "src/megatron/bridge/__init__.py",
    },
)
