# download_all_modified_libero_rlds.py
"""
下载 openvla/modified_libero_rlds 完整数据集
需要：pip install huggingface_hub[hf_transfer]
"""

import os
from huggingface_hub import snapshot_download


# 配置
repo_id = "lerobot/metaworld_mt50"
revision = "v3.0"                     # 目前还是 main 分支

# 保存路径（建议留足 15~20GB 空间）
local_save_path = "/path/to/datasets/metaworld_mt50"
# 强烈建议开启 hf_transfer 加速
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


print(f"开始下载完整数据集：{repo_id} ...")
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    local_dir=local_save_path,
    local_dir_use_symlinks=False,          # Windows 用户建议 False
    resume_download=True,
    # 如果你磁盘空间不够，也可以加 ignore_patterns 排除不需要的部分
    # ignore_patterns=["*/checkpoint*", "*.jsonl", "*.txt"],  
)

print(f"\n下载完成！全部文件保存在：{os.path.abspath(local_save_path)}")
# print("预计目录结构类似：")
# print(f"  {local_save_path}/")
# print("  ├── libero_spatial_no_noops/")
# print("  │   └── 1.0.0/")
# print("  │       ├── libero_spatial-train.tfrecord-00000-of-00016")
# print("  │       └── ...")
# print("  ├── libero_object_no_noops/")
# print("  ├── libero_goal_no_noops/")
# print("  └── libero_10_no_noops/")
