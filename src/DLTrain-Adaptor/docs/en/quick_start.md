# Quick Start

This document guides you through a quick hands-on experience of the core features of MegatronAdaptor and helps you successfully start Megatron-LM training on Ascend NPU devices.

> [!NOTE]
>
> The examples in this document are based on the matching versions of MegatronAdaptor core_r0.18.0 and Megatron-LM (Mcore) core_r0.18.0. For the matching version requirements of each component, see the [Release Notes](release_notes_ma.md).

## Environment Preparation

First, complete all steps in the [Installation Guide](install_guide.md), and use its "Installation Verification" section to confirm that MegatronAdaptor is installed successfully and the NPU device is available.

Obtain the Megatron-LM source code and switch to the matching version (core_v0.18.2):

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.18.2
```

> [!NOTE]
>
> Obtaining the Megatron-LM source code requires access to GitHub. If your network environment cannot smoothly access `github.com`, configure a proxy yourself, or use a Git mirror accessible from your region (such as GitCode or Gitee) to pull the source code. Then switch to the matching version `core_v0.18.2` and verify the repository address and branch to ensure that the source code is consistent with the matching version.

Before you start training, ensure that the CANN environment variables are loaded:

```bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## Importing the Adapter

Before you use Megatron-LM for training, you must import MegatronAdaptor in the Megatron-LM entry file.

**In the `Megatron-LM` directory, modify the `pretrain_gpt.py` file**, and add a line `import megatron_adaptor` after `import torch`:

```Python
# ... other imports
import torch
import megatron_adaptor # <-- Add this line
# ... other imports
```

If your data preprocessing script `tools/preprocess_data.py` also needs to call NPU operators, you must add `import megatron_adaptor` to it as well.

## Data Preparation (Optional, for Full Training)

If you only want to verify functionality, you can skip this step. To perform full pretraining, prepare the data as follows:

1. **Downloading the Tokenizer**
    Create the `Megatron-LM/gpt-tokenizer` directory, and download the `vocab.json` and `merges.txt` files from [HuggingFace](https://huggingface.co/Xenova/gpt-3.5-turbo/tree/main).

    <!-- > [!NOTE]
    >
    > Downloading the Tokenizer and the dataset requires access to HuggingFace. If your network environment cannot smoothly access `huggingface.co`, you are advised to download the same resources from a community site accessible from your region, such as [ModelScope](https://modelscope.cn). Pay attention to the correctness and security of the files to be downloaded. -->

2. **Downloading and converting the dataset**
    The following example uses the Alpaca dataset:

    ```python
    import pandas as pd
    import json

    # Download and read the data
    data_df = pd.read_parquet("train-00000-of-00001-a09b74b3ef9c3b56.parquet")
    data_df['text'] = data_df['text'].apply(lambda v: json.dumps({"text": v}))

    # Convert to the JSON format
    with open("alpaca_json.json", encoding='utf-8', mode='w') as f:
        for i, row in data_df.iterrows():
            f.write(row['text'])
            f.write('\n')
    ```

3. **Generating pretraining data**
    Use the preprocessing script that ships with Megatron-LM to convert the JSON data into the binary format:

    ```bash
    python tools/preprocess_data.py \
        --input alpaca_json.json \
        --output-prefix ./gpt_pretrain_data/alpaca \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file ./gpt-tokenizer/vocab.json \
        --merge-file ./gpt-tokenizer/merges.txt \
        --append-eod
    ```

## Starting Pretraining

In the `Megatron-LM` directory, create the pretraining script `train_distributed.sh`. The following is an example:

```bash
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
CKPT_DIR=./ckpt
VOCAB_FILE=<Specify path to file>/vocab.json
MERGE_FILE=<Specify path to file>/merges.txt
DATA_PATH=<Specify path and file prefix>_text_document
TP=2
PP=2
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
GPT_ARGS="
    --transformer-impl transformer_engine \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 1 \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 64 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 1000 \
    --init-method-std 0.01 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096.0 \
    --disable-bias-linear \
    --lr-warmup-fraction 0.01 \
    --fp16
"
DATA_ARGS="
    --split 990,5,5 \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
"
OUTPUT_ARGS="
    --log-throughput \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
```

Edit `train_distributed.sh` and configure the following paths:

```bash
CKPT_DIR=./ckpt
VOCAB_FILE=./gpt-tokenizer/vocab.json
MERGE_FILE=./gpt-tokenizer/merges.txt
DATA_PATH=./gpt_pretrain_data/alpaca_text_document
```

**Table 1**  train_distributed.sh parameter settings

|Parameter|Description|
|-|-|
|CKPT_DIR|Path to the weight files|
|VOCAB_FILE|Tokenizer file|
|MERGE_FILE|Tokenizer file|
|DATA_PATH|Dataset file|

> [!NOTE]
>
> - In the current example script, `NPUS_PER_NODE=8` means that 8 NPUs are required. If the number of NPUs you have is lower than this setting, reduce `NPUS_PER_NODE` accordingly and adjust the parallelism parameters such as `TP` and `PP` at the same time, which must satisfy `TP x PP ≤ NPUS_PER_NODE x NNODES`. Otherwise, you may encounter problems such as OOM.
> - Some parameters in the script, such as `--hidden-size`, `--num-layers`, and `--seq-length`, must be adapted to your actual scenario to avoid issues such as OOM.
> - By default, `--transformer-impl` is set to the recommended value `transformer_engine`, which is consistent with the repository CI scripts and provides Transformer acceleration based on TransformerEngineNPU. If you only need to verify the minimal runnable path and TransformerEngineNPU is not installed, you can change it to `local`, but Ascend Transformer acceleration operators will be unavailable.

Run the script to start pretraining:

```bash
bash ./train_distributed.sh
```

## Expected Output

After training starts, you will see log output similar to the following on the console:

```text
> initializing torch distributed ...
> compiling dataset index builder ...
> building GPT model ...
> building train, validation, and test datasets ...
> finished creating GPT datasets ...
done with setup ...
training ...
iteration        1/    1000 | consumed samples:            4 | elapsed time per iteration (ms): 3986.7 | throughput per GPU (TFLOP/s/GPU): 0.1 | learning rate: 1.000000E-07 | global batch size:     4 | lm loss: 1.041716E+01 | loss scale: 1.0 | grad norm: 15.144 | params norm: 136.026 | number of skipped iterations:   0 | number of nan iterations:   0 |Number of parameters in transformer block in billions:  0.05
iteration        2/    1000 | consumed samples:            8 | elapsed time per iteration (ms): 2173.0 | throughput per GPU (TFLOP/s/GPU): 0.2 | learning rate: 2.000000E-07 | global batch size:     4 | lm loss: 1.041835E+01 | loss scale: 1.0 | grad norm: 15.263 | params norm: 136.026 | number of skipped iterations:   0 | number of nan iterations:   0 |
...
```

## Stopping and Cleanup

**Stopping Training**:

- When training runs in the foreground, press `Ctrl+C` to stop training gracefully.
- If the training script has been suspended or runs in the background, use `kill <PID>` to terminate the process.

**Cleaning Up Training Artifacts**:

- Checkpoints: saved in the directory specified by `CKPT_DIR` in the script.
- Logs: usually printed to the console or redirected to a file. To clean them up, delete the log files directly.

> [!Troubleshooting]
>
> - **Issue**: An `Out of Memory (OOM)` message appears at runtime.
>   - **Solution**: Reduce the values of the `--micro-batch-size`, `--seq-length`, or `--hidden-size` parameters.
> - **Issue**: `torchrun` fails to start and reports `Address already in use`.
>   - **Solution**: Change the `MASTER_PORT` port number, such as 6002 or 6003.
> - **Issue**: The NPU device is not visible (`torch.npu.is_available()` returns `False`).
>   - **Solution**: Run the `npu-smi info` command to confirm whether the device is online, and check whether the CANN version matches the driver version. For details, see question Q2 in the FAQ of the [Installation Guide](install_guide.md).
> - **Issue**: A tokenizer or data path error is reported during data preprocessing or training startup.
>   - **Solution**: Confirm that `VOCAB_FILE`, `MERGE_FILE`, and `DATA_PATH` have been replaced with actual paths, and that `DATA_PATH` points to the `*_text_document` prefix (the binary data files are `<DATA_PATH>.bin` and `<DATA_PATH>.idx`).
> - **Issue**: The error `No module named 'megatron_adaptor'` is reported at startup.
>   - **Solution**: Confirm that you have completed the MegatronAdaptor installation as described in the installation guide, and that the CANN environment variables are loaded in the current Python environment (`source /usr/local/Ascend/cann/set_env.sh`).
> - **Issue**: Access to GitHub/HuggingFace times out or fails when you clone the Megatron-LM source code or download the Tokenizer and the dataset.
>   - **Solution**: This is a network environment restriction, not a configuration error. Configure a proxy yourself, or use channels accessible from your region, such as the GitCode or Gitee mirror to pull the source code, or a community site such as ModelScope to download the same resources. After you obtain resources from a third-party channel, be sure to verify the correctness and security of the repository address, branch, and files.
