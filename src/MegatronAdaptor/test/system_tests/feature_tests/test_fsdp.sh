#!/bin/bash
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# test_fsdp.sh - Megatron-FSDP (--use-megatron-fsdp) training on NPU
#
# Checkpointing (--save / --load) is deliberately NOT covered: fsdp_dtensor save/load trips
# an upstream Megatron-LM defect (the per-key metadata collectives assume every rank iterates
# the same optimizer-state keys, which the distributed optimizer does not guarantee),
# asserting or hanging the save.
#
# Constraints (--use-megatron-fsdp):
#   - PP must be 1: the dp-tp mesh requires dp_cp * ep * tp == world_size; no VPP
#   - CUDA_DEVICE_MAX_CONNECTIONS must be != 1
#   - --ckpt-format must be fsdp_dtensor (the only format Megatron-FSDP allows)
#   - optimizer limited to sgd/adam; DistributedOptimizer is forced

set -e

source "test/system_tests/env_npu.sh"
export STREAMS_PER_DEVICE=32

# --use-megatron-fsdp asserts CUDA_DEVICE_MAX_CONNECTIONS != 1
export CUDA_DEVICE_MAX_CONNECTIONS=2

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE * NNODES))

DATA_PATH="/home/dataset/llama2/alpaca_text_document"
TOKENIZER_MODEL="/home/dataset/model/llama-2-7b-hf/tokenizer.model"

TP=2
PP=1
CP=1
EP=1
DP=$((WORLD_SIZE / (TP * PP * CP)))

TRAIN_ITERS=10
MBS=4
GBS=32
SEQ_LENGTH=1024

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --expert-model-parallel-size ${EP}
"

GPT_ARGS="
    --num-layers 12
    --hidden-size 512
    --num-attention-heads 8
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --transformer-impl transformer_engine
    --attention-softmax-in-fp32
    --timing-log-level 0
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --make-vocab-size-divisible-by 1
"

TRAIN_ARGS="
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters ${TRAIN_ITERS}
    --lr 0.00015
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --lr-decay-iters 320000
    --lr-warmup-fraction .01
    --weight-decay 1e-2
    --clip-grad 1.0
    --seed 42
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH}
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1
    --eval-interval 1000
    --eval-iters 10
    --distributed-backend nccl
"

# --ckpt-format fsdp_dtensor is the only format Megatron-FSDP allows.
FSDP_ARGS="
    --use-megatron-fsdp
    --ckpt-format fsdp_dtensor
    --data-parallel-sharding-strategy optim_grads_params
    --use-distributed-optimizer
    --calculate-per-token-loss
    --use-mcore-models
"

# Upstream case: non-fused wgrad (--no-gradient-accumulation-fusion).
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${FSDP_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    --no-gradient-accumulation-fusion

# Fused wgrad (no --no-gradient-accumulation-fusion): exercises the te_general_gemm path
# under Megatron-FSDP.
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS} \
    ${TRAIN_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${FSDP_ARGS} \
    ${MODEL_PARALLEL_ARGS}
