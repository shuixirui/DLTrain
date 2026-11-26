# Developer Contribution Guide

Welcome to contribute to the TransformerEngineNPU project! This guide introduces the contribution workflow, code and testing standards, documentation requirements, and PR gating rules to help your changes get merged faster.

TransformerEngineNPU is an adaptation of the native [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) for Ascend devices. In addition to general engineering standards, it also has several "keep consistent with upstream" constraints. Please read this document in full before development.

## 1. Contribution Workflow

### 1.1 Feature Proposal and Review

Before starting to develop a new feature, follow the process below:

1. **Issue submission**: Submit an issue in the Issues section of the [GitCode](https://gitcode.com/Ascend/TransformerEngineNPU) repository, describing the feature you want to implement or the problem you want to fix.
2. **Proposal review**:
   - Huawei employees: Conduct the proposal review through the internal review group or review meetings (please confirm the specific group with the repository maintainers)
   - Community contributors: Conduct public discussion through the issue
3. **Proposal design**: Write a software proposal design document, use software engineering languages such as class diagrams and sequence diagrams to describe the overall solution and modification points, and clearly specify the behavioral differences from the upstream TransformerEngine.
4. **Development guidelines understanding**: Read this document carefully, follow the development guidelines, avoid introducing technical debt, and facilitate subsequent maintenance.

### 1.2 Development Workflow

```plaintext
Feature proposal → Solution review → Code development → Local self-verification → Documentation writing → PR submission → CI gate (comment `compile`) → Code review → Merge
```

## 2. Code Development Guidelines

### 2.1 Directory Responsibilities

| Directory | Responsibility |
| ---- | ---- |
| `transformer_engine/common/` | Framework-agnostic common modules, including quantization strategies (`recipe/`), etc. |
| `transformer_engine/pytorch/` | PyTorch backend core: `module/` (Linear, LayerNormLinear, LayerNormMLP, etc.), `tensor/` (FP8/MXFP8/HIF8/W4A8 and other quantized tensors), `recipe/`, `ops/`, `triton/` (triton-ascend kernel), `debug/` |
| `transformer_engine_torch/` | Python-side wrappers supporting the torch extension (data types, THD partition indices, etc.) |
| `tests/` | Test cases; see Chapter 3 for details |
| `docs/zh/` | Chinese documentation; `docs/en/` is maintained by the translation team |
| `build_tools/VERSION.txt` | Version number file, which must stay consistent with `setup.py` and `transformer_engine/__init__.py` |
| `pre-commit/` | Check configurations for ruff, pylint, bandit, typos, and gitleaks |
| `.gitcode/workflows/` | PR gate pipeline definitions and UT execution scripts |
| `ci/run_ci.sh` | Gate UT/ST scripts (mutually referenced with the pipeline scripts) |

### 2.2 Compatibility Constraints with Upstream TransformerEngine

- **Keep module names consistent**: The Python package name of this repository remains `transformer_engine`, the same as the native TransformerEngine. Do not install both in the same environment, and do not rename it for this reason.
- **Align public APIs with upstream**: The signatures, default values, and serialization formats of external interfaces should remain consistent with upstream. When differences are truly necessary, explain the reason in the documentation and annotate it with comments in the code.
- **Explicitly mark NPU-specific capabilities**: Ascend extension capabilities such as HIF8, W4A8, and W4A4-MXFP4 should be clearly marked as Ascend extensions in naming and documentation to avoid confusion with upstream semantics.
- **Respect capability boundaries**: TransformerEngineNPU does not support NVFP4 low-precision quantization. Do not add switches or branches for this purpose.
- **Keep commits atomic when syncing upstream**: Submit upstream synchronization and NPU adaptation changes separately. Do not include unrelated formatting or code rearrangement, so as to facilitate subsequent synchronization and issue tracing.

### 2.3 Code Design Principles

1. **Architectural decoupling**: Decouple NPU adaptation from the upstream implementation to facilitate keeping up with subsequent upstream version iterations.
2. **No duplicate implementation**: For capabilities already available upstream, prioritize reuse and adaptation for the NPU implementation, avoiding maintaining two parallel sets of semantics.
3. **Abstraction and reuse**: Prioritize extracting common capabilities. Extract logic as soon as it appears three times, avoiding large-scale copy-paste.
4. **Accurate naming**: Name functions and variables clearly and accurately. Vague expressions are prohibited.
5. **No monolithic structures**: God classes and god functions are prohibited.
6. **Cautious dynamic rewriting**: When using decorators or monkey patches, the original semantics must remain unchanged, and the necessity must be explained.

### 2.4 Code Quality Requirements

| Metric | Limit |
|------|------|
| Lines per function | ≤ 200 lines |
| Lines per file | ≤ 1000 lines |
| Naming conventions | Clear and accurate, avoid vague expressions |
| God class/function | Prohibited |

### 2.5 Compatibility Handling

- When incompatibility with existing capabilities is discovered, add code interception and provide a clear error message, and supplement the documentation with compatibility notes. Silent degradation is prohibited.

## 3. Test Case Guidelines

### 3.1 Test Case Location

| Case | Purpose |
| ---- | ---- |
| `tests/pytorch/` | PyTorch API UTs (mainly single-card, including the distributed test framework) |
| `tests/pytorch/tests_extend/` | Extended UTs (GEMM, GroupedLinear, optimizers, context parallelism, etc.) |
| `tests/pytorch/debug/` | Precision debugging capability UTs (executed separately in the gate) |
| `tests/atk/` | ATK precision comparison cases |
| `tests/agent/` | UTs for repository Agent workflow scripts |
| `tests/system_tests/` | System test (ST) scripts, requiring Megatron-LM + MegatronAdaptor collaboration |

For new cases, select the location based on the dimension of the module under test, and make the file name correspond to the object under test (e.g., `test_grouped_linear.py`).

### 3.2 Test Requirements

1. **Scenario coverage**: Ensure that test cases cover the main usage scenarios.
2. **Concise and complete**: Test cases should be concise but cover the necessary scenarios; more is not always better.
3. **Parameterized testing**: Use `pytest.mark.parametrize` to test multiple data types and input parameters.

### 3.3 Distributed Test Cases

- Distributed test cases can inherit the `DistributedTest` class in `tests/pytorch/distributed_testing.py`
- **Note**: Distributed initialization is time-consuming. Single-card test cases must not inherit `DistributedTest`

## 4. Documentation Guidelines

### 4.1 Documentation Location

For new features, place documentation in an appropriate path under the `docs` folder based on the feature dimension.

### 4.2 Documentation Content Requirements

The documentation must include the following content:

- **Problem analysis**: Clearly describe the problem being solved.
- **Solution**: Explain the technical approach in detail; do not summarize it in one or two sentences.
- **Usage scenarios**: Include usage restrictions and applicable conditions.
- **Usage instructions**: Provide code examples and configuration instructions.
- **Usage results**: Present performance or accuracy improvement data.

### 4.3 Documentation Format

- It is recommended to use software engineering languages such as class diagrams, activity diagrams, and sequence diagrams to describe the solution.
- If necessary, add compatibility notes between features.

## 5. Code Submission and PR Guidelines

### 5.1 PR Title and Description

- **Title**: Concise and complete, summarizing the main changes.
- **Description**: Clearly explain the purpose, content, and impact of the changes.
- **Test results**: New features must provide accuracy and performance test results.

### 5.2 CI Gate

- After submitting a PR, comment `compile` to trigger the CI gate check.
- Merging requires passing the gate and review by at least two developers (+2).

### 5.3 Branch Management

- Consider the branches involved. If necessary, submit PRs to both main and the corresponding branch.
- Pay attention to the release cadence and reserve time in advance for PR modifications and merging. Requests to merge first and modify later are not accepted.

### 5.4 Code Check

- Follow the coding standards. Code check suppressions must be reviewed line by line; batch or arbitrary suppression requests are prohibited.
- Submissions exceeding 500 lines require a code walkthrough and review.
- Strictly control quality, avoid low-level mistakes, and resolve issues as soon as they are found.

### 5.5 Open Source Compliance

- Check the open source license of any introduced open source code.
- Add the corresponding copyright notice to newly added files.

### 5.6 Commit Strategy

- **Small, incremental steps**: Avoid committing a large amount of code at once.
- **Quality first**: Ensure code quality in every commit.

## 6. Community Code of Conduct

### 6.1 Contributor Conventions

- Respect the work and opinions of others.
- Keep communication friendly and professional.
- Follow the project's code standards and best practices.
- Respond to feedback and issues in a timely manner.

### 6.2 Issue Feedback

- Use clear titles and detailed descriptions.
- Provide reproduction steps and environment information.
- Follow the issue template requirements.

## 7. Contact Us

- **GitHub/GitCode**: Communicate through issues and PRs.
- **Huawei employees**: Search for `MindSpeed TMG Group` on WeLink.
- **Community discussion**: Join the project's Discussions section.

Thank you to all developers for your contributions! The development of TransformerEngineNPU would not be possible without the efforts of every developer.

*TransformerEngineNPU Team*
