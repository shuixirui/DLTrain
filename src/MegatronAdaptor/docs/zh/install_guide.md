# 软件安装

本文档旨在指导用户从零开始，基于PyTorch框架快速、正确地完成MegatronAdaptor的安装。

## 硬件与操作系统要求

**表 1** 产品硬件支持列表

|产品|是否支持（训练场景）|
|--|:-:|
|<term>Ascend 950PR&950DT系列产品</term>|√|
|<term>Atlas A3训练系列产品</term>|√|
|<term>Atlas A3推理系列产品</term>|x|
|<term>Atlas A2训练系列产品</term>|√|
|<term>Atlas A2推理系列产品</term>|x|
|<term>Atlas 200I/500 A2推理产品</term>|x|
|<term>Atlas推理系列产品</term>|x|
|<term>Atlas训练系列产品</term>|x|

> [!NOTE]
>
> 本节表格中“√”代表支持，“x”代表不支持。

- 各硬件产品对应物理机部署场景支持的操作系统请参考[兼容性查询助手](https://www.hiascend.com/hardware/compatibility)。

- 各硬件产品对应虚拟机及容器部署场景支持的操作系统请参考《CANN 软件安装》的“[操作系统兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)”章节。

## 安装前准备

请参见《版本说明》中的“[相关产品版本配套说明](release_notes_ma.md#相关产品版本配套说明)”章节，下载安装对应的软件版本。

请单击[固件与驱动](https://hiascend.com/hardware/firmware-drivers)，并根据引导完成固件与驱动的安装。

> [!NOTE]
>
> 安装运行程序建议使用非root用户，且建议对安装程序的目录文件做好权限管控：文件夹权限设置为750，文件权限设置为640。可以通过设置umask控制安装后文件的权限，如设置umask为0027。
> 更多安全相关内容请参见《[安全声明](SECURITYNOTE.md)》中各组件关于“文件权限控制”的说明。

## 安装MegatronAdaptor

1. 安装CANN

   安装配套版本的NPU驱动固件、CANN软件（Toolkit、ops 和 NNAL）并配置环境变量，具体请参考《[CANN 软件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920/softwareinst/instg/instg_0000.html)》。

   CANN软件提供进程级环境变量设置脚本，训练或推理场景下使用NPU执行业务代码前需要调用该脚本，否则业务代码将无法执行。

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

2. 安装PyTorch以及TorchNPU

   请参考《[TorchNPU 快速安装](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C)》，获取配套版本的PyTorch以及TorchNPU软件包。

   > [!NOTE]
   >
   > - 如环境中存在旧版本MegatronAdaptor，请先[卸载](#卸载)旧版本，再安装新版本。
   >
   > - 更多TorchNPU插件版本请单击[Link](https://gitcode.com/ascend/pytorch/releases)。

3. 源码安装

   ```shell
   git clone https://gitcode.com/Ascend/MegatronAdaptor.git
   cd MegatronAdaptor
   git checkout core_r0.18.0
   pip install -e .
   cd ..
   ```

4. 安装TransformerEngineNPU

   Megatron-LM在昇腾设备上的Transformer加速能力由TransformerEngineNPU提供（[TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)），需拉取源码安装：

   ```shell
   git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
   cd TransformerEngineNPU
   git checkout 26.2.0_release_v2.17
   pip install -e . --no-build-isolation
   cd ..
   ```

   > [!WARNING]
   >
   > TransformerEngineNPU和原生TransformerEngine的库模块名相同，为避免冲突，请确保同一环境内不要同时安装两者，保留TransformerEngineNPU即可。

5. 获取Megatron-LM源码

   具体操作如下所示：

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_v0.18.2
   cd ..
   ```

   > [!NOTE]
   >
   > - 获取Megatron-LM源码需访问GitHub。若您所处的网络环境无法顺畅访问 `github.com`，请自行配置代理，或使用国内可访问的Git镜像（如GitCode、Gitee等）拉取源码。
   > - 拉取前请核对仓库地址与分支或者commit，确保所获取的Megatron-LM源码与配套版本一致，并关注其正确性与安全性。

6. 安装Triton-Ascend

   安装配套版本的Triton-Ascend，请参考《[Triton-Ascend安装指南](https://triton-ascend.readthedocs.io/zh-cn/latest/installation_guide.html#piptriton-ascend)》，获取配套版本的Triton-Ascend安装指令。

   可参考如下安装命令：

   ```shell
   # 注意：triton-ascend 3.2.0及以下版本，Triton-Ascend和Triton不能同时存在。需要先卸载社区Triton，再安装Triton-Ascend。
   pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

## 卸载

如需卸载MegatronAdaptor，执行以下命令：

```shell
pip uninstall -y megatron_adaptor
```
