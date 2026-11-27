import json
import os
import sys
from pathlib import Path

model_path = os.environ.get("MODEL_PATH", "/mnt/afs-a3-weight-share/Intern-S2-Preview")

import torch
from transformers.utils.import_utils import is_flash_linear_attention_available

print("HF_FLA_BEFORE_ADAPTOR:",
      is_flash_linear_attention_available(), flush=True)
from transformers import Qwen3NextForCausalLM
import transformers
from transformers.dynamic_module_utils import get_class_from_dynamic_module

InternS2HF = get_class_from_dynamic_module(
    "modeling_interns2_preview.InternS2PreviewForConditionalGeneration",
    model_path,
    local_files_only=True,
)
transformers.InternS2PreviewForConditionalGeneration = InternS2HF
print("HF_CLASS_REGISTERED:", InternS2HF.__module__, flush=True)

import megatron_adaptor

workdir = Path(os.environ["WORKDIR"])
sys.path.insert(0, str(workdir / "src/DLTrain-Bridge/src"))

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_bridge import Qwen35VLMoEBridge
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import Qwen35VLMoEModelProvider

@MegatronModelBridge.register_bridge(
    source=InternS2HF,
    target=Qwen3VLModel,
    provider=Qwen35VLMoEModelProvider,
    model_type="intern_s2_preview",
)
class InternS2PreviewBridge(Qwen35VLMoEBridge):
    pass


from transformers import AutoConfig
from megatron.bridge.models.conversion import auto_bridge as ab_module

hf_config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)
lookup_key = ab_module.get_causal_lm_class_name_via_auto_map(
    config=hf_config,
)
assert lookup_key == "InternS2PreviewForCausalLM", repr(lookup_key)

@MegatronModelBridge.register_bridge(
    source=lookup_key,
    target=Qwen3VLModel,
    provider=Qwen35VLMoEModelProvider,
    model_type="intern_s2_preview",
)
class InternS2PreviewCausalBridge(InternS2PreviewBridge):
    pass

print("BRIDGE_LOOKUP_KEY:", lookup_key, flush=True)

bridge = AutoBridge.from_hf_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
)
provider = bridge.to_megatron_provider(load_weights=True)

settings = {
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 1,
    "expert_model_parallel_size": 8,
    "expert_tensor_parallel_size": 2,
    "context_parallel_size": 1,
    "sequence_parallel": True,
    "params_dtype": torch.bfloat16,
    "pipeline_dtype": torch.bfloat16,
    "bf16": True,
    "fp16": False,
    "seq_length": 128,
    "language_max_sequence_length": 128,
    "kv_channels": 256,
    "mtp_num_layers": 0,
    "freeze_vision_model": True,
    "freeze_vision_projection": True,
    "moe_permute_fusion": False,
    "gradient_accumulation_fusion": False,
}
for name, value in settings.items():
    setattr(provider, name, value)

expected = {
    "num_layers": 40,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_query_groups": 2,
    "num_moe_experts": 256,
    "moe_router_topk": 8,
}
for name, value in expected.items():
    actual = getattr(provider, name)
    assert actual == value, (name, actual, value)

provider.finalize()
fields = list(expected) + list(settings) + [
    "experimental_attention_variant", "linear_attention_freq",
    "moe_ffn_hidden_size",
]
summary = {name: getattr(provider, name, None) for name in fields}
text = json.dumps(summary, indent=2, default=str)
print(text, flush=True)
(workdir / "logs").mkdir(parents=True, exist_ok=True)
(workdir / "logs/interns2-provider.json").write_text(
    text + "\n", encoding="utf-8",
)
print("INTERNS2_PROVIDER_CONFIG_OK", flush=True)
