# 交付与维护工具

- `verify_sources.py`：检查源码入口、Core链接及Git记录的源码变化。
- `audit_git_changes.py`：采集单仓库commit、状态、源码摘要和运行脚本，生成交付审计ZIP；同时保留对早期分仓快照的核查能力。
- `copy_adamw_result_to_workspace.sh`：将结果包放到服务器/root/workspace以便下载。
- `handoff/`：WebIDE终端使用的代码传输命令，内嵌内容与校验和须同步生成。
- `legacy/`：早期修复的维护工具。修复已合入src，正常运行不需要执行这些工具。

环境与运行准备入口是`train/environment/prepare_workspace.py`，不是本目录。报告与测试入口见benchmark/。
