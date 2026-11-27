# Release Notes

## Key Features

MegatronAdaptor built-in features cover basic runtime environment adaptation, basic Megatron adaptation, NPU Profiling, NPU data dump precision comparison, NPU deterministic computation, and more.

## Version Mapping

### Product Version

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>Product</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MegatronAdaptor</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>26.2.0_core_r0.18.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>Release version</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Release Date</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>September 2026</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>Maintenance</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6 months</p>
</td>
</tr>
</tbody>
</table>

### Related Product Versions

**Table 1** MegatronAdaptor software version mapping

|MegatronAdaptor Code Branch Name|CANN Version|TorchNPU Version|Python Version|PyTorch Version|Triton-Ascend Version|TransformerEngineNPU Version|
|--|--|--|--|--|--|--|
|26.2.0_core_r0.18.0|9.2.X|26.2.X|Python3.12|2.10.0|3.6.0|26.2.0_release_v2.17|

>[!NOTE]
>
> - You can choose the MegatronAdaptor code branch as needed to download the source code and install it.
>
> - The Triton-Ascend version is strongly bound to the CANN version. The Triton-Ascend version must correspond to the CANN version one-to-one. For details, see [Triton-Ascend Compatibility](https://triton-ascend.readthedocs.io/en/latest/release_note.html#id13).

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates incompatibility and "Y" indicates compatibility.

**Table 2** MegatronAdaptor and TorchNPU version compatibility

<table style="table-layout: fixed; width: 531px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="3">MegatronAdaptor</th>
    <th colspan="1">TorchNPU Version</th>
  </tr>
  <tr>
    <th>26.2.X</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td>26.2.0_core_r0.18.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 3** MegatronAdaptor and CANN version compatibility

<table style="table-layout: fixed; width: 547px"><colgroup>
<col style="width: 162px">
<col style="width: 91px">
<col style="width: 94px">
<col style="width: 100px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th rowspan="3">MegatronAdaptor</th>
    <th colspan="1">CANN Version</th>
  </tr>
  <tr>
    <th>9.2.X</th>
  </tr></thead>
<tbody>
    <tr>
    <td>26.2.0_core_r0.18.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## Important Notes

The library module name of TransformerEngineNPU is the same as that of the native TransformerEngine. To avoid conflicts, do not install both in the same environment. Keep only TransformerEngineNPU.

## Update Notes

### New Features

The built-in features cover basic runtime environment adaptation, basic Megatron adaptation, NPU Profiling, NPU data dump precision comparison, NPU deterministic computation, and more.

### Deleted Features

None.

### Interface Change Notes

None.

### Resolved Issues

None.

### Known Issues

None.

## Upgrade Impact

### Impact on the System During the Upgrade

- Service impact.

    Upgrading the software version interrupts service.

- Network communication impact.

    It has no impact on communication.

### Impact on the System After the Upgrade

None.

## Documentation

|Document|Summary|Update Notes|
|--|--|--|
|[MegatronAdaptor Quick Start](quick_start.md)|Describes how MegatronAdaptor enables Megatron-LM to run efficiently on Ascend devices.|-|
|[MegatronAdaptor Installation Guide](install_guide.md)|Guides you through installing MegatronAdaptor on an NPU based on the PyTorch framework. It covers hardware and operating system compatibility, and the complete installation process for drivers, firmware, and CANN base software. It helps you quickly build a distributed LLM training environment.|-|

## Virus Scan Results and Vulnerability Patch List

### Virus Scan Results

|Antivirus Software|Antivirus Software Version|Virus Database Version|Scan Time|Scan Result|
|--|--|--|--|--|
|QiAnXin|8.0.5.5260|2026-10-07 08:00:00.0|2026-10-08|No viruses or malware|
|Kaspersky|12.0.0.6672|2026-10-08 10:11:00|2026-10-08|No viruses or malware|
|Bitdefender|7.5.1.200224|7.101645|2026-10-08|No viruses or malware|

### Fixed Vulnerabilities

None.

## Revision History

|Document Version|Release Date|Change Description|
|--|--|--|
|01|2026-10-15|First official release.|
