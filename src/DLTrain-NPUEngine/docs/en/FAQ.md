# FAQs

This document describes the capability boundary of TransformerEngineNPU (TENPU) regarding NVFP4 support.

## NVFP4 Support

### Symptom

When using the NVFP4 recipe (such as `recipe.NVFP4BlockScaling`) or migrating the NVFP4 quantization configuration from a GPU training script to the Ascend environment, quantization does not take effect or an unsupported error is reported.

### Root Cause

TENPU currently does not support NVFP4 low-precision quantization, and `is_nvfp4_available()` always returns `False`. This limitation is a capability boundary of the software stack and is unrelated to version compatibility.

### Solution

Use other recipes on Ascend (such as `DelayedScaling`, `Float8BlockScaling`, `MXFP8BlockScaling`, etc.; for the complete list, see [API List](api/api_list.md#fp8low-precision-training)).
