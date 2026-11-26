# DLTrain

面向昇腾的 Megatron 分布式训练适配与验证代码。当前以 Intern-S2-Preview 35B 的纯文本路径验证环境、原生 HF 权重加载、分布式优化器更新和训练日志。本仓库完整包含四个组件的固定版本源码，普通 clone 无需初始化 submodule 或另行 clone 依赖。

## 源码组织

| 目录 | 内容 | 上游基准提交 |
| --- | --- | --- |
| `src/Megatron-LM/` | Megatron-Core训练、并行与优化器 | `ba7b5ebce12af60627a80985792a1449ce45f46c` |
| `src/MegatronAdaptor/` | 昇腾适配器 | `59653dff2df7bf03b528a7fb8a7f1fb8a288f741` |
| `src/TransformerEngineNPU-2.17/` | 昇腾Transformer Engine实现 | `09f6cc1a17314112bc1cdeef1e061c724f256f38` |
| `src/Megatron-Bridge-interns2/` | HF权重转换与Provider，包含4个文件的适配修改 | `a069bdffa41fdc4d7cb377e72a89d484167b8a85` |
| `cluster/` | 注册、加载、训练、探针、数据处理、启动与报告工具 | 本项目代码 |
| `requirements/` | 集群实际使用的依赖约束 | 部分为范围约束，非完整环境锁文件 |
| `patches/bridge/` | 继承的2文件改动及我们新增的2文件修复 | 已应用于当前源码 |
| `tools/` | 部署准备及源码完整性核查 | 本项目代码 |

[来源版本清单](sources.lock.json)、[文件摘要](sources.manifest.json)、[改动说明](SOURCE_CHANGES.md)和[第三方许可说明](THIRD_PARTY_NOTICES.md)共同记录源码来源。三个上游仓库相对固定提交无修改；Bridge的249个服务器盘点文件与本仓库对应文件逐字节一致。

## 获取与使用

```bash
git clone https://github.com/shuixirui/DLTrain.git
cd DLTrain
python3 tools/verify_sources.py
```

实测环境是 Linux ARM64、Ascend A3 / 910C、16个逻辑NPU，Python3.12.13、CANN9.1.0镜像、PyTorch2.10.0、torch-npu2.10.0.post7、TENPU2.17.1、Triton-Ascend3.6.0.post2和Transformers5.2.0。源码合并不包含驱动、CANN、Python虚拟环境、模型权重或语料；安装和运行仍需对应硬件与依赖。

现有集群启动脚本使用工作目录顶层的训练入口。将本仓库放在原实测路径后，执行下列准备命令会复制所需入口与requirements到顶层，保留脚本与服务器版本一致：

```bash
python3 tools/prepare_workspace.py
```

环境安装、原生tokenizer语料准备、路径调整和启动命令见[部署说明](docs/DEPLOYMENT.md)。准备命令不启动训练、不安装依赖、不复制权重。

## 已验证范围

Intern-S2-Preview 35B纯文本路径已完成16进程权重加载和独立500步训练，使用MCore原生分布式AdamW、BF16、TP2 / PP1 / EP8 / ETP2 / CP1 / DP8。SEQ64 / GBS8和SEQ64 / GBS32分别消耗256,000和1,024,000目标token。已抽检embedding和第0层本地专家分片，检查所有rank首步FP32主权重、动量与BF16回写。

这些记录证明运行与抽样更新检查通过。MoE不等长reduce_scatter提示、GDN权重覆盖和AccumulateGrad stream提示仍需专项数值对照。当前关闭MTP、冻结视觉，无训练checkpoint保存或持出集评估；指定起始权重revision身份、DLRL正式调用和联合RL导出部署仍待验证。

本地纯Python检查可用以下命令执行，NPU训练需要集群环境：

```bash
python3 -m unittest discover -s cluster -p 'test_adamw_*.py'
python3 -m unittest discover -s cluster -p 'test_expert_etp_numpy.py'
```

第三方源码保留各自许可证。模型、数据集及其使用条件独立于代码交付。
