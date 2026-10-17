import tensorflow as tf
import os
from tqdm import tqdm

def check_tfrecords(data_dir):
    bad_files = []
    # 递归查找所有 tfrecord 文件
    tfrecord_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if "tfrecord" in file:
                tfrecord_files.append(os.path.join(root, file))

    print(f"找到 {len(tfrecord_files)} 个文件，开始扫描...")

    for file_path in tqdm(tfrecord_files):
        try:
            # 尝试迭代读取文件中的每一条数据
            dataset = tf.data.TFRecordDataset(file_path)
            for _ in dataset:
                pass 
        except Exception as e:
            print(f"\n[!!!] 发现损坏文件: {file_path}")
            print(f"错误信息: {e}")
            bad_files.append(file_path)

    return bad_files

if __name__ == "__main__":
    # 将此路径替换为你实际的数据集根目录
    DATASET_PATH = "/path/to/rlds/dataset/0.1.0"
    corrupted = check_tfrecords(DATASET_PATH)

    if corrupted:
        print("\n扫描完成。以下文件已损坏，建议删除：")
        for f in corrupted:
            print(f)
    else:
        print("\n扫描完成，未发现损坏文件。")
