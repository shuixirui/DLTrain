# 环境与训练

实测软件组合：Linux ARM64、Python3.12.13、CANN9.1.0镜像、PyTorch2.10.0、torch-npu2.10.0.post7、Triton-Ascend3.6.0.post2、TENPU2.17.1、Transformers5.2.0、NumPy1.26.4和PyArrow24.0.0。训练使用16个逻辑NPU。依赖约束在environment/requirements/；其中部分为范围约束，不等同于完整环境锁文件或镜像摘要。

## 安装环境

开发机需预先提供昇腾驱动、CANN、Python3.12及C++编译器。此前实测使用平台镜像`registry2.d.pjlab.org.cn/ccr-lepton-official-images/a3-cann:9.1.0-a3-ubuntu22.04-py3.12`；镜像标签不等于镜像摘要。安装脚本不安装系统驱动或CANN。

在仓库根目录执行：

```bash
bash install.sh
```

默认使用`/usr/local/python3.12.13/bin/python3.12`创建`.venv-dltrain`，安装依赖和三个editable源码包，准备运行入口，再检查NPU、Bridge/ModelOpt导入及四个组件的实际源码路径。默认不需要模型权重。使用`bash install.sh --dry-run`可只预览命令，不创建环境或下载依赖。

| 环境变量 | 默认值或用途 |
| --- | --- |
| VENV_PATH | 当前仓库的.venv-dltrain；运行examples/run.sh时需指定相同路径 |
| PYTHON_BIN | /usr/local/python3.12.13/bin/python3.12 |
| CANN_ENV_FILE | /usr/local/Ascend/cann/set_env.sh |
| ATB_ENV_FILE | /usr/local/Ascend/nnal/atb/set_env.sh，存在时加载 |
| PIP_INDEX_URL | 华为云PyPI镜像 |
| PIP_EXTRA_INDEX_URL | 华为云Ascend软件源 |
| MODEL_PATH | --check-provider时使用的本地模型目录 |

可选Provider配置检查及已有隔离环境复用：

```bash
VENV_PATH="$PWD/.venv-clean-test" \
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview \
bash install.sh --check-provider
```

成功时输出`DLTRAIN_IMPORT_CHECK_OK`和`DLTRAIN_CLEAN_ENV_READY`；开启Provider检查后还须出现`INTERNS2_PROVIDER_CONFIG_OK`。日志和实际包版本写入`logs/install-*.log`及对应的`*-packages.txt`。Git目录记录当前提交；ZIP目录使用已有的`source-commit.txt`，没有记录时会明确标注，安装不依赖Git。

2026-11-26，提交`239cde340cc8db7b7d67303aebf69a6a390ccac5`的ZIP源码在原开发机的新目录及`.venv-clean-test`中，按上述依赖和源码安装命令通过了隔离环境`pip check`及Provider检查。这是手动命令的验证记录；新`install.sh`入口尚待服务器执行和全新开发机复验。

CANN环境脚本会将系统工具包加入PYTHONPATH，因此安装阶段可能报告`ms-service-profiler`等外部工具缺依赖或版本冲突。安装器用`python -I -m pip check`单独检查虚拟环境，并继续执行真实导入检查；这不代表CANN profiler等附加功能已验证。HF模型的docstring报错、FLA可选探测和弃用提示也应保留在日志中；实际训练仍需单独验证。

## 源码安装方式

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
VENV_PATH="$PWD/.venv-dltrain" \
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
