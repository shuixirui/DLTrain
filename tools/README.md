# 环境与验证

tools只保留安装与20步训练需要的12个功能文件：environment中7个，validation中5个。训练入口位于顶层examples/，分词实现在src/DLTrain-framework/tools/prepare_interns2_corpus.py。

## 环境安装

实测组合为Linux ARM64、Python3.12.13、CANN9.1.0镜像、PyTorch2.10.0、torch-npu2.10.0.post7、Triton-Ascend3.6.0.post2、TENPU2.17.1、Transformers5.2.0、NumPy1.26.4及PyArrow24.0.0。训练使用16个逻辑NPU。此前平台镜像为`registry2.d.pjlab.org.cn/ccr-lepton-official-images/a3-cann:9.1.0-a3-ubuntu22.04-py3.12`；该标签不等于镜像摘要。

开发机需提供驱动、CANN、Python3.12及C++编译器。仓库根目录执行：

```bash
bash install.sh
```

默认创建仓库内`.venv-dltrain`，安装依赖和三个editable源码包，检查组件路径、Bridge/ModelOpt导入及NPU。Bridge通过源码路径加载，不单独重解其依赖。ModelOpt固定0.44.0，原生tokenizer使用sentencepiece0.2.1。

| 环境变量 | 默认值或用途 |
| --- | --- |
| VENV_PATH | 仓库内.venv-dltrain；示例运行使用相同路径 |
| PYTHON_BIN | /usr/local/python3.12.13/bin/python3.12 |
| CANN_ENV_FILE | /usr/local/Ascend/cann/set_env.sh |
| ATB_ENV_FILE | /usr/local/Ascend/nnal/atb/set_env.sh，存在时加载 |
| PIP_INDEX_URL | 华为云PyPI镜像 |
| PIP_EXTRA_INDEX_URL | 华为云Ascend软件源 |
| MODEL_PATH | 可选Provider检查所用的本地权重目录 |

```bash
bash install.sh --dry-run
VENV_PATH="$PWD/.venv-clean-test" \
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview \
bash install.sh --check-provider
```

默认安装不需要模型权重。成功输出`DLTRAIN_IMPORT_CHECK_OK`和`DLTRAIN_CLEAN_ENV_READY`；启用Provider检查后还应有`INTERNS2_PROVIDER_CONFIG_OK`。日志和包版本保存为`logs/install-*.log`及对应的`*-packages.txt`。Git目录记录提交；ZIP目录读取已有source-commit.txt，没有记录时会明确标注。

environment/requirements/中的部分依赖仍为范围约束，不等同于完整环境锁文件。CANN加入PYTHONPATH的附加工具可能在安装时报告缺依赖；隔离虚拟环境检查采用`python -I -m pip check`，并执行实际导入检查。这不代表CANN profiler等附加功能已验证。日志中的FLA探测、模型docstring和弃用提示应保留。

environment/prepare_workspace.py检查当前文件布局及Bridge到Core的链接，并写入组件基准提交记录；不会再复制顶层脚本。它不会安装依赖或启动NPU任务。

## 验证模块

| 文件 | 用途 |
| --- | --- |
| validation/check_interns2_provider.py | 注册HF模型和Bridge，生成并检查Provider配置，也是训练初始化的一部分 |
| validation/adamw_dist_update_probe.py | 验证首步FP32主权重、AdamW一二阶矩及BF16回写 |
| validation/adamw_shard_layout.py | 确定当前rank拥有的优化器参数分片和抽检点 |
| validation/adamw_math.py | 学习率调度及首步AdamW参考公式 |
| validation/optimizer_update_probe.py | 更新验证所需的标量及梯度抽检辅助函数 |

## 运行与记录

```bash
VENV_PATH="$PWD/.venv-dltrain" \
MODEL_PATH=/mnt/afs-a3-weight-share/Intern-S2-Preview \
bash examples/run.sh
```

默认20步、SEQ64、MBS1、DP8和梯度累积4次，得到GBS32；TP2/PP1/EP8/ETP2/CP1、BF16。分布式AdamW峰值LR1e-5，沿用500步余弦调度和20步warmup。可用TRAIN_STEPS调整步数。

启动脚本验证data/en_head.json.gz的SHA256，调用framework内的原生tokenizer分词脚本，随后执行examples/train_interns2.py。训练及验证代码直接从所在目录加载。日志写入logs/，指标、配置、源码快照及首步检查写入runs/；MTP关闭、视觉冻结，不保存训练checkpoint。

239cde3的ZIP源码在原开发机的新目录及.venv-clean-test中通过手动安装、Provider和20步训练检查。当前目录迁移、本安装入口、全新开发机以及DLRL集成仍需相应实机验证。
