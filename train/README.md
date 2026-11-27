# 昇腾实测环境与启动

源码包含在本仓库中；Python包、CANN、驱动、权重与语料需要部署环境提供。集群成功组合为Python3.12.13、CANN9.1.0镜像、torch2.10.0、torch-npu2.10.0.post7、Triton-Ascend3.6.0.post2、TENPU2.17.1、Transformers5.2.0、NumPy1.26.4和PyArrow24.0.0。requirements保留集群当时的约束，部分包未锁死，不能把它当作完整镜像摘要或可确定性重建承诺。

## 路径与环境

原实测工作目录为 `/mnt/afs-ptdata-shared/sxr/dltrain-mcore018`，模型为 `/mnt/afs-a3-weight-share/Intern-S2-Preview`，虚拟环境为工作目录下`.venv`。现有脚本保留这两个实际绝对路径。更换目录需同步检查Provider、加载、训练和启动脚本的引用；不要直接在另一目录误启动旧工作区。

在该工作目录检出本仓库后，先运行：

```bash
python3 tools/verify_sources.py
python3 train/environment/prepare_workspace.py
```

prepare_workspace从train、data和benchmark收集运行资源到工作目录顶层，并按environment/component_versions.py为四个组件生成source-commit.txt，使训练入口仍能记录组件基准版本。已有文件不同则在写入前拒绝覆盖；它不修改组件源码，不启动任务。DLTrain自己的修改版本以Git提交为准。

在Linux ARM64建立Python3.12虚拟环境，按requirements中的runtime/adaptor/model/bridge-compat/tokenizer约束安装依赖。模型加载需要ModelOpt distill/Megatron插件；成功实测使用nvidia-modelopt0.44.0。源码包使用现有依赖进行editable安装，防止自动重解整套适配栈：

```bash
python -m pip install --no-deps --no-build-isolation -e src/DLTrain-TransformerEngineNPU-2.17
python -m pip install --no-deps --no-build-isolation -e src/DLTrain-Megatron-LM
python -m pip install --no-deps --no-build-isolation -e src/DLTrain-MegatronAdaptor
```

以上是源码安装部分，不能替代前置依赖安装和CANN环境配置。Bridge由Provider脚本通过src路径加载，不单独pip安装以重解依赖。MegatronAdaptor导入时执行其上游自带运行时适配，Core源码未改动不等于没有运行时补丁。

## 权重与语料

使用原生Intern-S2 tokenizer，额外需要sentencepiece0.2.1。配置脚本先做HF可选FLA探测，再导入适配器；HF_FLA_BEFORE_ADAPTOR=False是可选库探测结果，不等于模型不含GDN。

公开语料实测为RedPajama V2英文子集，原生tokenizer预处理后保存到`datasets/interns2-redpajama-500/tokens.bin`和`tokens.meta.json`。GBS32启动器检查至少1,024,001 token；实测缓存2,000,000 token。目录不随源码发布，下载及预处理流程见data中的RedPajama和prepare_interns2_corpus脚本。

当前共享模型可以做功能验证，但其指定起始revision `4f57cab513689b089019fce4ad24e26520df183c` 的文件身份尚待正式核对。加载成功不能代替revision一致性证明。

## 启动已验证配方

准备好环境、原生权重与语料，并确认16个逻辑NPU可用后，在子Bash中执行：

```bash
bash <<'BASH'
set -eo pipefail
WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
export ADAMW_JOB_ID="interns2-redpajama-adamw-gbs32-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p /root/workspace
nohup bash "$WORKDIR/run_interns2_adamw_gbs32.sh" \
  > "/root/workspace/$ADAMW_JOB_ID.log" 2>&1 < /dev/null &
printf 'STARTED_PID=%s\nLOG=/root/workspace/%s.log\n' "$!" "$ADAMW_JOB_ID"
BASH
```

该原样配方先独立验证3步，再从原始权重开始500步；不是DLRL联调入口。SEQ64、MBS1、DP8、梯度累积4得到GBS32；TP2/PP1/EP8/ETP2/CP1，BF16，AdamW峰值LR1e-5、warmup20和cosine，clip=1。更短步数应另存启动配置并保留日志，不能把当前500步脚本描述为默认20步。

项目联调仍需课题二按实际DLRL接口调用DLTrain、加载约定起始权重、记录一次真实RL优化器更新及至少一项参数变化，再完成导出交接。源码合并没有替代这些验收。
