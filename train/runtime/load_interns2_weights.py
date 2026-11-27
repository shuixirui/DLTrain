import json
import os
import runpy
from datetime import timedelta
from pathlib import Path

import torch
import torch_npu
import torch.distributed as dist
from safetensors import safe_open

workdir = Path(os.environ["WORKDIR"])
model_dir = Path(os.environ.get("MODEL_PATH", "/mnt/afs-a3-weight-share/Intern-S2-Preview"))
local_rank = int(os.environ["LOCAL_RANK"])

torch.npu.set_device(local_rank)
dist.init_process_group("hccl", timeout=timedelta(minutes=10))
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 16, f"Expected 16 ranks, got {world}"

state = runpy.run_path(str(workdir / "check_interns2_provider.py"))
provider = state["provider"]
provider.initialize_model_parallel(seed=42)

print(f"rank={rank}: START_REAL_WEIGHT_LOAD", flush=True)
models = provider.provide_distributed_model(wrap_with_ddp=False)
if not isinstance(models, (list, tuple)):
    models = [models]

index = json.loads(
    (model_dir / "model.safetensors.index.json").read_text()
)
key = "model.language_model.embed_tokens.weight"
shard = model_dir / index["weight_map"][key]

verified = 0
decoder_found = False
for model in models:
    for name, module in model.named_modules():
        if name.endswith("language_model.decoder"):
            assert len(module.layers) == 40, "实际模型层数错误"
            decoder_found = True

        if name.endswith("language_model.embedding.word_embeddings"):
            weight = module.weight
            assert weight.device.type == "npu", str(weight.device)
            start = int(module.vocab_start_index)

            with safe_open(str(shard), framework="pt", device="cpu") as f:
                expected = f.get_slice(key)[start:start + 4]

            actual = weight.detach()[:4].cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            verified += 1
            print(
                f"rank={rank}: EMBEDDING_EXACT_MATCH "
                f"start={start} dtype={weight.dtype}",
                flush=True,
            )

assert decoder_found, "未找到语言模型 decoder"
assert verified > 0, "未找到语言模型 embedding"

torch.npu.synchronize()
ok = torch.tensor([1.0], device=f"npu:{local_rank}")
dist.all_reduce(ok)
assert ok.item() == world

if rank == 0:
    print("INTERNS2_REAL_WEIGHT_LOAD_CHECK_OK", flush=True)

dist.destroy_process_group()
