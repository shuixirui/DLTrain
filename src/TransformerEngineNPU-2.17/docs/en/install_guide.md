# TransformerEngineNPU Installation Guide

This document describes how to install TransformerEngineNPU (TENPU) and its supporting software from source code.

## Supported Hardware and Operating Systems

**Table 1** Supported hardware list

|Product|Supported (Training Scenario)|
|--|:-:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 training products</term>|√|
|<term>Atlas A3 inference products</term>|x|
|<term>Atlas A2 training products</term>|√|
|<term>Atlas A2 inference products</term>|x|
|<term>Atlas 200I/500 A2 inference products</term>|x|
|<term>Atlas inference products</term>|x|
|<term>Atlas training products</term>|x|

> [!NOTE]
>
> In the table in this section, "√" indicates supported, and "x" indicates not supported.

<!-- - For the operating systems supported by each hardware product in physical machine deployment scenarios, refer to the [Compatibility Checker](https://www.hiascend.com/hardware/compatibility).

- For the operating systems supported by each hardware product in virtual machine and container deployment scenarios, refer to the "[Operating System Compatibility Notes](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum)" section in *CANN Software Installation*. -->

## Preparation Before Installation

See the "[Related Product Version Matching](release_notes_tenpu.md#version-compatibility-of-related-products)" section in *Release Notes* to download and install the corresponding software version.

Click [Firmware and Drivers](https://hiascend.com/en/hardware/firmware-drivers) and follow the instructions to complete the installation of the firmware and drivers.

> [!NOTE]
>
> It is recommended to use a non-root user to install and run programs, and to properly control the permissions of the installation program directory files: set the folder permission to 750 and the file permission to 640. You can control the permissions of installed files by setting umask, for example, setting umask to 0027.
>
> For more security-related content, see the description of "File Permission Control" for each component in *[TransformerEngineNPU Security Statement](SECURITYNOTE.md)*.

## Installing TransformerEngineNPU

TransformerEngineNPU is a Transformer acceleration library for Ascend devices. It can be installed independently and used directly in custom training scenarios. If you train with Megatron-LM, you also need to install MegatronAdaptor and obtain the Megatron-LM source code.

1. Install CANN

   Install the matching NPU driver and firmware and CANN software (Toolkit), and configure the CANN environment variables. For details, see *[CANN Quick Installation](https://www.hiascend.com/en/cann/download)*.

   The CANN software provides a process-level environment variable setting script. Before using the NPU to execute service code in training or inference scenarios, you need to invoke this script; otherwise, the service code cannot be executed.

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   The preceding command uses the default installation path as an example. Replace it with the actual path of `set_env.sh`.

2. Install PyTorch and TorchNPU

   See *[TorchNPU Quick Installation](https://www.hiascend.com/en/developer/software/ai-frameworks/pytorch/download)* to obtain the matching PyTorch and TorchNPU software packages.

   > [!NOTE]
   >
   > - If an earlier version of TransformerEngineNPU exists in the environment, [uninstall](#uninstalling-transformerenginenpu) the earlier version before installing the new version.
   >
   > - For more TorchNPU plugin versions, click [Link](https://gitcode.com/ascend/pytorch/releases).

3. Install Triton-Ascend

   Refer to *[Triton-Ascend Installation Guide](https://triton-ascend.readthedocs.io/en/latest/installation_guide.html#id1)* to obtain the installation commands for the matching Triton-Ascend version.

   You can refer to the following installation command:

   ```shell
   pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

4. Install TransformerEngineNPU

   TransformerEngineNPU provides Transformer acceleration operators on Ascend devices and must be installed from source code:

   ```shell
   git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
   cd TransformerEngineNPU
   git checkout 26.2.0_release_v2.17
   pip install -e . --no-build-isolation
   cd ..
   ```

   > [!NOTE]
   >
   > TransformerEngineNPU and the native TransformerEngine share the same library module name. To avoid conflicts, ensure that both are not installed in the same environment; keep only TransformerEngineNPU.

5. Install MegatronAdaptor (Optional)

   The typical usage scenario of TransformerEngineNPU is to train with Megatron-LM. In this case, you also need to install MegatronAdaptor and obtain the Megatron-LM source code.

   ```shell
   git clone https://gitcode.com/Ascend/MegatronAdaptor.git
   cd MegatronAdaptor
   git checkout 26.2.0_core_r0.18.0
   pip install -e .
   cd ..
   ```

6. Obtain the Megatron-LM source code (Optional)

   The specific operations are as follows:

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_r0.18.0
   cd ..
   ```

   > [!NOTE]
   >
   > Obtaining the Megatron-LM source code requires access to GitHub. If your network environment cannot smoothly access `github.com`, configure a proxy by yourself, or use a Git mirror accessible (such as GitCode or Gitee) to pull the source code, and then switch to the matching branch `core_r0.18.0`.

## Verifying the Installation

After the installation is complete, perform the following checks in sequence to confirm that the environment is available.

1. Verify that the NPU is available

   ```shell
   python -c "import torch; import torch_npu; print(torch.npu.is_available()); print(torch.npu.device_count())"
   ```

   The expected output is `True` and the visible number of NPUs.

2. Verify the import and version of TransformerEngineNPU

   ```shell
   python -c "import transformer_engine; print(transformer_engine.__version__)"
   ```

   The expected output is `2.17.1`. If a native version was previously installed in the environment, run `python -c "import transformer_engine; print(transformer_engine.__file__)"` to confirm that the import path points to the TransformerEngineNPU source code directory.

3. Verify FP8 availability

   ```shell
   python -c "from transformer_engine.pytorch import is_fp8_available; print(is_fp8_available())"
   ```

   The expected output is `True`. If `False` is returned, pass the `return_reason=True` parameter to obtain the specific reason.

## Uninstalling TransformerEngineNPU

To uninstall TransformerEngineNPU, run the following command:

```shell
pip uninstall -y transformer_engine
```
