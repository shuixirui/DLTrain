# 源码来源与实际改动

本次合并以服务器审计包 `dltrain-git-audit-20261126T180946Z-469704.zip` 为依据。三个Git组件的HEAD均为sources.lock.json中固定提交，status、相对HEAD和基准的diff为空，没有基准之后的新提交。

## Bridge的四处文件改动

| 文件，相对Bridge源码根目录 | 来源 | 内容 |
| --- | --- | --- |
| `src/megatron/bridge/models/conversion/param_mapping.py` | 共享快照原有，8增2删 | AutoMapping增加MindSpeed命名的列/行并行模块、PTNorm识别，以及融合LayerNorm/Linear和TELinear的兼容识别；不表示本项目安装了MindSpeed |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/model.py` | 共享快照原有，4增4删 | 注释四处CUDA NVTX性能标记调用 |
| `src/megatron/bridge/models/model_provider.py` | 本次适配，4增2删 | 两处旧ModelType.encoder_and_decoder比较增加成员存在性保护，兼容MCore0.18；添加补丁标记 |
| `src/megatron/bridge/models/qwen_vl/qwen3_vl_bridge.py` | 本次适配，39增25删 | 专家gate/up和down映射在ETP切分前按完整权重形状检查；非主rank交给原映射通信，主rank转换目标dtype；gate/up分别切分后配对 |

当前完整源码已应用三份补丁，运行时不应重复应用。补丁只用于复现或审查。将固定上游基准加上继承补丁、ModelType补丁和专家ETP补丁，可重建服务器四个修改文件；重建结果与服务器当前文件逐字节一致。

审计工具原始difflib输出在一个文件末尾缺失换行时把删除/新增行连在一起。本仓库派生补丁补上标准Git的“No newline at end of file”标记，经git apply检查、反向还原、备份SHA256/原始Git blob校验和逐文件重放验证。原始审计包没有改写。

## 仓库外新增的训练代码

服务器工作目录顶层共有28个Python、shell和requirements文件，已分别纳入cluster/与requirements/。核对过这些文件的实际SHA256；load_interns2_weights.py等早期脚本以服务器版本归档，没有用本地未部署候选代替。

新增代码包括Intern-S2模型注册、Provider配置、真实权重加载与分片抽检、SGD/AdamW训练入口、更新探针、原生分布式优化器配置、语料预处理、启动、运行归档与绘图。分布式AdamW通过DDP配置和OptimizerConfig启用Core自带实现，没有修改Megatron核心优化器。

本地另有未部署的ModelOpt按需导入补丁草稿，不属于实际适配修改，未作为正式源码补丁纳入。历史上传工具、旧配方和本地验证工具也保留于cluster/，其存在不表示都部署到服务器；默认参考分布式AdamW和GBS32入口。

## 源码完整性边界

四组件完整上游源码按固定提交导入，Bridge再应用上述适配补丁。Bridge服务器审计范围内249个文件及28个脚本/依赖文件均有摘要核对。完整组件源码还包含上游文档和测试，它们不代表已在NPU执行。

Bridge上游的`3rdparty/Megatron-LM`原为指向`3cbe5c68a40c5820c11fc60fea1143dc6983706e`的Git子模块。本仓库将其改为指向同仓库已固定MCore0.18源码的相对符号链接，避免嵌套子模块和重复版本；原始引用及替换记录在sources.lock.json的integration_links中。这是源码整合布局变更，位于服务器249文件核对范围之外。

本仓库不携带任何上游.git目录；上游历史通过仓库地址和提交号定位。Linux clone保留5个上游符号链接及上述1个整合链接，Windows检出使用Git的core.symlinks=false形式。文件哈希清单可用于检查源码内容，不能证明运行中的包、外部权重或容器镜像身份。
