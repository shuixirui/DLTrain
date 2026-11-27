# 实测、报告与本地检查

本目录集中管理日志解析、绘图、告警清点和不需要NPU的检查。训练入口及环境准备位于tools/，分词位于tools/preprocess/，data/仅保存原始数据与说明。

在仓库根目录运行：

```bash
python -m unittest discover -s tools/benchmark/tests -v
python tools/benchmark/plot_interns2_log.py /path/to/launcher.log --output-dir results/report
python tools/benchmark/plot_interns2_log.py /path/to/launcher.log --last-step 200 --output-dir results/first200
python tools/benchmark/audit_interns2_warnings.py /path/to/launcher.log --output results/warning-audit.json
```

绘图需要NumPy和Matplotlib。图表及CSV按日志打印精度生成，正式500步与3步验证阶段分开，完整性检查失败时不会补点或伪造通过状态。移动平均、梯度对数轴及耗时预热范围均在图中标注。

tests覆盖优化器数学、分布式分片探针、启动协调器、实际源码的ETP布局、压缩语料解析、示例预览、测量记录、工作目录准备和日志完整性。源日志不随源码发布；依赖特定真实日志的检查在文件缺失时明确跳过。这里的CPU和模拟通信检查不代表已执行真实NPU通信或联合RL验收。
