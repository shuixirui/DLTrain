# Software Installation

This document explains how to install MegatronAdaptor quickly and correctly from scratch based on the PyTorch framework.

## Hardware and OS Requirements

**Table 1** Product hardware support list

| Product | Supported (Training Scenario) |
|--|:-:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 training products</term> | √ |
| <term>Atlas A3 inference products</term> | x |
| <term>Atlas A2 training products</term> | √ |
| <term>Atlas A2 inference products</term> | x |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term> | x |
| <term>Atlas training products</term> | x |

> [!NOTE]
>
> The "√" in the table indicates support, and "x" indicates no support.

<!-- - For the OSs supported by each hardware product in physical machine deployment scenarios, see the [Compatibility Query Assistant](https://www.hiascend.com/hardware/compatibility). -->

For the OSs supported by each hardware product in VM and container deployment scenarios, see [OS Compatibility](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/920/softwareinst/instg/instg_0101.html?OS=openEuler&InstallType=netyum) in *CANN Software Installation*.

## Preparation Before Installation

See [Related Product Versions](release_notes_ma.md#related-product-versions) in *Release Notes* to download and install the corresponding software version.

Click [Firmware and Drivers](https://hiascend.com/hardware/firmware-drivers) and follow the guidance to install the firmware and drivers.

> [!NOTE]
>
> You are advised to install and run the program as a non-root user, and you are also advised to control the permissions on the installation program directory and files: set the folder permissions to 750 and the file permissions to 640. You can control the permissions on the files after installation by setting `umask`, for example, setting `umask` to 0027.
> For more security-related information, see the description of "file permission control" for each component in the [Security Statement](SECURITYNOTE.md).

## Installing MegatronAdaptor

1. Install CANN.

   Install the matching versions of the NPU driver, firmware, and CANN software (Toolkit, ops, and NNAL), and configure the environment variables. For details, see [CANN Software Installation](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/920/softwareinst/instg/instg_0000.html).

   The CANN software provides a process-level environment variable setting script. Before you use an NPU to run service code in training or inference scenarios, call this script. Otherwise, the service code cannot run.

   ```shell
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

2. Install PyTorch and TorchNPU.

   See [TorchNPU Quick Installation](https://www.hiascend.com/en-sg/developer/software/ai-frameworks/pytorch/download?versionId=180&ids=89dda9ba9de741349efa03687a487678%2C204%2C200%2C1%2C6%2C177%2C) to obtain the matching versions of the PyTorch and TorchNPU software packages.

   > [!NOTE]
   >
   > - If an old version of MegatronAdaptor exists in the environment, [uninstall](#uninstallation) the old version first, and then install the new version.
   >
   > - For more TorchNPU plugin versions, click [Link](https://gitcode.com/ascend/pytorch/releases).

3. Install from source.

   ```shell
   git clone https://gitcode.com/Ascend/MegatronAdaptor.git
   cd MegatronAdaptor
   git checkout 26.2.0_core_r0.18.0
   pip install -e .
   cd ..
   ```

4. Install TransformerEngineNPU.

   On Ascend devices, Transformer acceleration for Megatron-LM is provided by TransformerEngineNPU ([TransformerEngineNPU](https://gitcode.com/Ascend/TransformerEngineNPU)), which you need to install from source:

   ```shell
   git clone https://gitcode.com/Ascend/TransformerEngineNPU.git
   cd TransformerEngineNPU
   git checkout 26.2.0_release_v2.17
   pip install -e . --no-build-isolation
   cd ..
   ```

   > [!WARNING]
   >
   > TransformerEngineNPU and the native TransformerEngine share the same library module name. To avoid conflicts, ensure that you do not install both in the same environment. Keep only TransformerEngineNPU installed.

5. Obtain the Megatron-LM source code.

   The steps are as follows:

   ```shell
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   git checkout core_v0.18.2
   cd ..
   ```

   > [!NOTE]
   >
   > - Obtaining the Megatron-LM source code requires access to GitHub. If your network environment cannot smoothly access `github.com`, configure a proxy yourself, or use a Git mirror accessible from your region (such as GitCode or Gitee) to pull the source code, and then switch to the matching branch.
   > - Before pulling the source code, verify the repository address and branch/commit to ensure that the obtained Megatron-LM source code is consistent with the matching version, and pay attention to its correctness and security.

6. Install Triton-Ascend.

   Install the matching version of Triton-Ascend. See the [Triton-Ascend Installation Guide](https://triton-ascend.readthedocs.io/en/latest/installation_guide.html#piptriton-ascend) to obtain the installation command for the matching version of Triton-Ascend.

   You can use the following installation command:

   ```shell
   # Note: For triton-ascend 3.2.0 and earlier, Triton-Ascend and Triton cannot coexist. Uninstall the community Triton before installing Triton-Ascend
   pip install triton-ascend==3.6.0 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
   ```

## Uninstallation

To uninstall MegatronAdaptor, run the following command:

```shell
pip uninstall -y megatron_adaptor
```
