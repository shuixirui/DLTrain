# Release Notes

## Key Features

TransformerEngineNPU is an adaptation of the native TransformerEngine for Ascend devices, providing low-precision computation acceleration for large model training and inference. It provides high-performance foundational layer modules such as `Linear`, `GroupedLinear`, `LayerNormLinear`, `LayerNormMLP`, and `DotProductAttention`, as well as the `autocast` quantization context. It supports low-precision strategies such as FP8 (delayed/current/block-wise scaling), MXFP8, HIF8, W4A8, and W4A4-MXFP4, along with features such as MoE routing fusion, CPU Offload, Swap context-parallel attention, and the `FusedAdam` optimizer.

## Version Compatibility Description

### Product Version Information

| Item | Description |
|------|------|
| Product Name | MindSpeed |
| Product Version | 26.2.0 |
| Version Type | Release version |
| Component Name | TransformerEngineNPU |
| Release Date | October 2026 |
| Maintenance Period | 6 months |

### Version Compatibility of Related Products

**Table 1** TransformerEngineNPU software version compatibility table

|TransformerEngineNPU Code Branch Name|MegatronAdaptor Version|CANN Version|TorchNPU Version|Python Version|PyTorch Version|Triton-Ascend Version|
|--|--|--|--|--|--|--|
|26.2.0_release_v2.17|26.2.0_core_r0.18.0|9.2.0|26.2.0|Python3.12|2.10.0|3.6.0|

>[!NOTE]
>
> - Users can select the TransformerEngineNPU code branch as needed to download the source code and install it.
> - The Triton-Ascend version is strongly bound to the CANN version. The use of Triton-Ascend should correspond one-to-one with the CANN version. For details, see [Triton-Ascend Compatibility](https://triton-ascend.readthedocs.io/en/latest/release_note.html#id13).

**Table 2** Version correspondence between TransformerEngineNPU and native TransformerEngine

| TransformerEngineNPU | TransformerEngine |
|----------------------|-------------------|
| 26.2.0_release_v2.17 | release_v2.17 |

### Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates incompatibility, and "Y" indicates compatibility.

**Table 3** TransformerEngineNPU and TorchNPU version compatibility

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">TransformerEngineNPU</th>
      <th colspan="1">TorchNPU Version</th>
    </tr>
    <tr>
      <th>26.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.X_release_v2.17</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**Table 4** TransformerEngineNPU and CANN version compatibility

<table style="table-layout: fixed; width: 750px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">TransformerEngineNPU</th>
      <th colspan="1">CANN Version</th>
    </tr>
    <tr>
      <th>9.2.X</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>26.2.X_release_v2.17</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## Version Usage Notes

- TransformerEngineNPU and the native TransformerEngine share the same library module name. To avoid conflicts, do not install both in the same environment. Keep only TransformerEngineNPU.
- TransformerEngineNPU does not support NVFP4 low-precision quantization.

## Update Notes

### New Features

- Adapted to MegatronAdaptor and Mcore 0.18.0.
- Attention: Added Swap context parallel attention and the corresponding Swap Flash Attention implementation.
- MoE: Added MoE routing fusion interfaces (`fused_topk_with_score_function`, `fused_moe_aux_loss`, etc.); `GroupedLinear` supports weight gradient fusion accumulation (`fuse_wgrad_accumulation`); refactored grouped matrix multiplication unified scheduling.
- Low-precision training: Added HIF8 independent delayed scaling strategy, W4A8 weight quantization strategy, and W4A4-MXFP4; enhanced block-level FP8 capabilities, supporting sharded weight requantization and batched master weight requantization; supported quantized tensor checkpoint serialization; added the `CustomRecipe` custom strategy; added fused Triton kernels for FP8 amax/scale updates.
- Optimizer: `FusedAdam` supports delayed FP8 and mixed moment updates, and supports quantized parameters and BF16 residuals.
- Basic modules: Refactored `LayerNormLinear`/`LayerNormMLP` to the `WeightGradStore` delayed weight gradient architecture; supports `gradient_accumulation_fusion`; supports `backward_override` and `save_original_input`.
- Debugging: Aligned the precision debugging capabilities with the upstream TransformerEngine.

### Removed Features

None

### Interface Change Notes

None

### Resolved Issues

- Fixed the MXFP8 training memory leak issue.
- Fixed the issue where block-level FP8 precision became inconsistent after checkpoint recovery.
- Fixed the error of `LayerNormMLP` and `Linear` in scenarios without `main_grad`.
- Fixed the issue with the `FusedAdam` group step when `state_dict` is not an int.

### Known Issues

None

## Upgrade Impact

### Impact on the Current System During the Upgrade

- Impact on services

    TransformerEngineNPU is installed from source code. A version upgrade requires rebuilding and reinstalling it and restarting the training task, during which services are interrupted.

- Impact on network communication

    There is no impact on communication.

### Impact on the Current System After Upgrade

None

## Related Documents

|Document Name|Content Description|Update Notes|
|--|--|--|
|*[TransformerEngineNPU Introduction](introduction.md)*|Introduces the purpose, functional features, and directory structure of TransformerEngineNPU.|-|
|*[TransformerEngineNPU Installation Guide](install_guide.md)*|Guides you through installing TransformerEngineNPU and its companion software from source code, covering hardware and OS compatibility, and the complete process of installing basic software such as drivers, firmware, and CANN.|-|
|*[TransformerEngineNPU API List](api/api_list.md)*|Lists all public APIs, compares them with the native TransformerEngine, and provides environment variable descriptions.|-|
|*[TransformerEngineNPU FAQs](FAQ.md)*|Describes the capability boundary issues of NVFP4.|-|

## Virus Scan Result

|Antivirus Software Name|Antivirus Software Version|Virus Definition Version|Scan Time|Scan Result|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-10-07 08:00:00.0|2026-10-08|Virus-free and Malware-free|
|Kaspersky|12.0.0.6672|2026-10-08 10:11:00|2026-10-08|Virus-free and Malware-free|
|Bitdefender|7.5.1.200224|7.101645|2026-10-08|Virus-free and Malware-free|

## Vulnerability Patch List

None

## Revision Notes
 
| Document Version | Release Date | Revision Notes |
| --- | --- | --- |
| 01 | 2026-10-15 | First official release. |
