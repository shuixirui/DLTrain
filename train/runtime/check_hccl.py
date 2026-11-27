import os
from datetime import timedelta

import torch
import torch_npu
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group("hccl", timeout=timedelta(seconds=120))

rank = dist.get_rank()
world = dist.get_world_size()
expected = world * (world + 1) // 2

for dtype in (torch.float32, torch.bfloat16):
    x = torch.full(
        (4096,), rank + 1,
        dtype=dtype, device=f"npu:{local_rank}",
    )
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    assert (x == expected).all().item(), (
        f"rank={rank}, dtype={dtype}, 求和结果错误"
    )

torch.npu.synchronize()
print(f"rank={rank}/{world}: FP32_BF16_ALLREDUCE_OK", flush=True)
dist.barrier()
if rank == 0:
    print(f"HCCL_{world}_DEVICES_OK", flush=True)
dist.destroy_process_group()
