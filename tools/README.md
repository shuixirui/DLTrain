# 环境与训练

实测软件组合：Linux ARM64、Python3.12.13、CANN9.1.0镜像、PyTorch2.10.0、torch-npu2.10.0.post7、Triton-Ascend3.6.0.post2、TENPU2.17.1、Transformers5.2.0、NumPy1.26.4和PyArrow24.0.0。训练使用16个逻辑NPU。依赖约束在environment/requirements/；其中部分为范围约束，不等同于完整环境锁文件或镜像摘要。

## 安装源码

准备Python3.12虚拟环境、兼容驱动/CANN及依赖后，在仓库根目录执行源码安装：

```bash
python -m pip install --no-deps --no-build-isolation -e src/DLTrain-NPUEngine
python -m pip install --no-deps --no-build-isolation -e src/DLTrain-framework
python -m pip install --no-deps --no-build-isolation -e src/DLTrain-Adaptor
```

Bridge通过Provider中的源码路径加载，不单独安装以重解依赖。ModelOpt distill/Megatron插件需可用，原实测安装nvidia-modelopt0.44.0；原生tokenizer还需要sentencepiece0.2.1。已有editable安装在组件目录改名后，需要使用上述新路径重新安装。

## 运行示例

```bash
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview \
VENV_PATH=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018/.venv \
bash examples/run.sh
```

启动脚本以当前仓库为WORKDIR，默认20步、SEQ64、MBS1、DP8、梯度累积4，即GBS32；TP2/PP1/EP8/ETP2/CP1，BF16。优化器为Core原生分布式AdamW，峰值LR1e-5，沿用500步余弦调度及20步warmup，示例只执行前20步。

脚本读取data/en_head.json.gz，先验证SHA256，然后使用原生tokenizer生成datasets/interns2-redpajama-raw缓存。该独立目录用于区分早期筛选JSONL生成的缓存。预处理逻辑位于preprocess/prepare_interns2_corpus.py，支持逐行JSON及gzip输入。

environment/prepare_workspace.py检查源码入口与Bridge的Core链接，随后将运行资源复制到仓库顶层，以适配现有训练入口；发现同名文件内容不同会停止。它还按component_versions.py记录组件基准版本，不安装依赖或启动NPU任务。

运行日志写入logs/<RUN_ID>.log，逐步测量、配置、摘要和更新探针结果写入runs/<RUN_ID>/。数据与模型权重路径可通过MODEL_PATH、CORPUS_DIR等环境变量调整。脚本参数和预览命令见[示例说明](../examples/README.md)。

## 已有训练入口

- runtime/：Provider注册、真实权重加载、训练循环及优化器更新探针。
- launch/：早期独立SGD、AdamW及3步验证后500步测量的协调器，保留其原实测工作目录配置。
- preprocess/：原生tokenizer语料准备。

新示例默认20步；launch中的原500步协调器不是默认示例。MTP关闭、视觉冻结，当前流程不保存训练checkpoint。指定起始权重revision的文件身份及DLRL真实调用、联合RL参数更新和导出仍须专项留证。
