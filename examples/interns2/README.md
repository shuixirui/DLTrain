# Intern-S2-Preview训练示例

先按[训练使用说明](../../train/README.md)准备Linux ARM64昇腾环境、原生权重和RedPajama token缓存，再在仓库根目录执行：

```bash
python3 tools/verify_sources.py
python3 train/environment/prepare_workspace.py --dry-run
python3 train/environment/prepare_workspace.py
```

prepare_workspace将train、data和benchmark中的运行资源整理到既有集群脚本使用的工作目录顶层，并写入组件基准记录；发现已有文件内容不同会停止，避免覆盖工作区修改。

现有启动器保留原实测工作目录`/mnt/afs-ptdata-shared/sxr/dltrain-mcore018`和权重目录`/mnt/afs-a3-weight-share/Intern-S2-Preview`。确认这两个位置后，从该工作目录使用生成的`run_interns2_adamw_gbs32.sh`启动。完整后台命令见训练使用说明。

GBS32实测配方为SEQ64、MBS1、DP8、梯度累积4、TP2/PP1/EP8/ETP2/CP1、BF16、原生分布式AdamW。启动器先检查3步，再从原始权重独立测量500步；它不是默认20步配方，也不是DLRL联调入口。

运行记录按RUN_ID保存，日志可写入/root/workspace下载；报告使用benchmark目录中的工具生成。MTP关闭，视觉冻结，未做持出集质量验收或checkpoint导出验收。
