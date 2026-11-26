# 版本说明

## 关键特性

MegatronAdaptor内置特性覆盖基础运行环境适配、Megatron基础适配、NPU Profiling、NPU数据Dump精度对照、NPU确定性计算等。

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="26.25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p><span>MegatronAdaptor</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>core_r0.18.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>release版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>发布时间</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>2026年10月</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="26.25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="73.75%"><p>6个月</p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

**表 1** MegatronAdaptor软件版本配套表

|MegatronAdaptor代码分支名称|CANN版本|TorchNPU版本|Python版本|PyTorch版本|Triton-Ascend版本|TransformerEngineNPU版本|
|--|--|--|--|--|--|--|
|core_r0.18.0|9.2.X|26.2.X|Python3.12|2.10.0|3.6.0|release_v2.17|
|26.2.0_core_r0.18.0|9.2.X|26.2.X|Python3.12|2.10.0|3.6.0|26.2.0_release_v2.17|

> [!NOTE]
>
>用户可根据需要选择MegatronAdaptor代码分支下载源码并进行安装。

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2** MegatronAdaptor与TorchNPU版本兼容

<table style="table-layout: fixed; width: 531px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">MegatronAdaptor</th>
    <th colspan="1">TorchNPU版本<br/></th>
  </tr>
  <tr>
  <th>26.2.X</th>
  </tr>
</thead>
<tbody>
    <tr>
    <td>core_r0.18.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**表 3** MegatronAdaptor与CANN版本兼容

<table style="table-layout: fixed; width: 547px"><colgroup>
<col style="width: 162px">
<col style="width: 91px">
<col style="width: 94px">
<col style="width: 100px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">MegatronAdaptor</th>
    <th colspan="1">CANN版本<br/></th>
  </tr>
  <tr>
  <th>9.2.X</th>
  </tr>
  </thead>
<tbody>
    <tr>
    <td>core_r0.18.0</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项

TransformerEngineNPU 和原生 TransformerEngine 的库模块名相同，为避免冲突，同一环境内请勿同时安装两者，仅保留 TransformerEngineNPU 即可。

## 更新说明

### 新增特性

内置特性覆盖基础运行环境适配、Megatron 基础适配、NPU Profiling、NPU 数据 Dump 精度对照、NPU 确定性计算等。

### 删除特性

无

### 接口变更说明

无

### 已解决问题

无

### 遗留问题

无

## 升级影响

### 升级过程中对现行系统的影响

- 对业务的影响

    软件版本升级过程中会导致业务中断。

- 对网络通信的影响

    对通信无影响。

### 升级后对现行系统的影响

无

## 配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《[MegatronAdaptor快速入门](quick_start.md)》|介绍基于MegatronAdaptor如何实现Megatron-LM在昇腾设备上的高效运行。|-|
|《[MegatronAdaptor安装指导](install_guide.md)》|指导如何在NPU上基于PyTorch框架完成MegatronAdaptor的安装，内容涵盖硬件与操作系统兼容性说明、驱动固件及CANN基础软件的完整安装流程，帮助用户快速搭建大模型分布式训练环境。|-|

## 病毒扫描及漏洞修补列表

### 病毒扫描结果

|防病毒软件名称|防病毒软件版本|病毒库版本|扫描时间|扫描结果|
|---|---|---|---|---|
|QiAnXin|8.0.5.5260|2026-10-07 08:00:00.0|2026-10-08|无病毒，无恶意|
|Kaspersky|12.0.0.6672|2026-10-08 10:11:00|2026-10-08|无病毒，无恶意|
|Bitdefender|7.5.1.200224|7.101645|2026-10-08|无病毒，无恶意|

### 漏洞修补列表

无

## 修订记录

|文档版本|发布日期|修改说明|
|--|--|--|
|01|2026-10-15|第一次正式发布|
