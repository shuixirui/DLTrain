# RedPajama原始数据

本目录保存此前训练使用的RedPajama V2原始sample分片，文件保持上游字节内容：

| 项目 | 内容 |
| --- | --- |
| 文件 | `en_head.json.gz` |
| 数据集 | [togethercomputer/RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) |
| 固定版本 | `949be9cdd1d82fbc379de4909dc4e89ca9b87680` |
| 上游路径 | `sample/documents/2023-06/0000/en_head.json.gz` |
| 字节数 | 56,818,623 |
| SHA-256 | `bc742de614bc508778a42a3d1d2f3cb71e558af88114a0c3fb9a67c6356e5ca7` |
| 格式 | gzip压缩的逐行JSON，正文在`raw_content`字段 |

这是单个英文head分片。数据来源、授权和原始网页信息见官方数据集说明及记录中的来源字段。

分词逻辑位于`train/preprocess/prepare_interns2_corpus.py`；示例启动脚本会从本原始文件生成独立token缓存。早期训练曾先筛选JSONL再分词，示例当前直接读取原始分片，缓存签名会与早期筛选版本不同。
