# import os
# import re
# import shutil
# from tqdm import tqdm
# from PIL import Image, ImageFile

# # 允许加载损坏的图像
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None

# def make_market_dir(dst_dir='./'):
#     train_path = os.path.join(dst_dir, 'bounding_box_train')
#     query_path = os.path.join(dst_dir, 'query')
#     test_path = os.path.join(dst_dir, 'bounding_box_test')

#     if not os.path.exists(train_path):
#         os.makedirs(train_path)
#     if not os.path.exists(query_path):
#         os.makedirs(query_path)
#     if not os.path.exists(test_path):
#         os.makedirs(test_path)

# def extract_mars_to_market(mars_path, market_path):
#     pattern = re.compile(r'(\d+)C(\d)T(\d+)F(\d+)\.jpg')
#     train_files = []
#     test_files = []
#     query_files = {}

#     for root, _, files in os.walk(mars_path):
#         for file in tqdm(files, desc="Processing MARS dataset", unit="file"):
#             if file.endswith('.jpg'):
#                 match = pattern.match(file)
#                 if match:
#                     pid, cam, track, frame = match.groups()
#                     pid = int(pid)
#                     cam = int(cam)
#                     track = int(track)
#                     frame = int(frame)

#                     # 生成 Market-1501 格式的文件名
#                     new_filename = f"{pid:04d}_c{cam}s{track}_{frame:06d}_00.jpg"

#                     if 'bbox_train' in root:
#                         train_files.append((os.path.join(root, file), new_filename))
#                     elif 'bbox_test' in root:
#                         test_files.append((os.path.join(root, file), new_filename))
#                         if pid not in query_files:
#                             query_files[pid] = []
#                         query_files[pid].append((os.path.join(root, file), new_filename))

#     # 将所有训练文件复制到 Market-1501 结构的 bounding_box_train 文件夹中
#     train_dst_dir = os.path.join(market_path, 'bounding_box_train')
#     for src, dst in tqdm(train_files, desc="Copying training files", unit="file"):
#         shutil.copy(src, os.path.join(train_dst_dir, dst))

#     # 将所有测试文件复制到 Market-1501 结构的 bounding_box_test 文件夹中
#     test_dst_dir = os.path.join(market_path, 'bounding_box_test')
#     for src, dst in tqdm(test_files, desc="Copying test files", unit="file"):
#         shutil.copy(src, os.path.join(test_dst_dir, dst))

#     # 将每个 PID 的第一张测试文件复制到 query 文件夹中
#     query_dst_dir = os.path.join(market_path, 'query')
#     for pid in query_files:
#         src, dst = query_files[pid][0]
#         shutil.copy(src, os.path.join(query_dst_dir, dst))

# if __name__ == '__main__':
#     mars_dataset_path = 'mars'  # 修改为 MARS 数据集的实际路径
#     market1501_output_path = 'market1501'  # 修改为目标 Market-1501 数据集路径

#     make_market_dir(market1501_output_path)
#     extract_mars_to_market(mars_dataset_path, market1501_output_path)

import os
import re
import shutil
from tqdm import tqdm
from PIL import Image, ImageFile
import random

# 允许加载损坏的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def make_market_dir(dst_dir='./'):
    train_path = os.path.join(dst_dir, 'bounding_box_train')
    query_path = os.path.join(dst_dir, 'query')
    test_path = os.path.join(dst_dir, 'bounding_box_test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(query_path):
        os.makedirs(query_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

def extract_mars_to_market(mars_path, market_path, num_images_per_id=25):
    pattern = re.compile(r'(\d+)C(\d)T(\d+)F(\d+)\.jpg')
    train_files = []
    test_files = []
    query_files = {}

    for root, _, files in os.walk(mars_path):
        for file in tqdm(files, desc="Processing MARS dataset", unit="file"):
            if file.endswith('.jpg'):
                match = pattern.match(file)
                if match:
                    pid, cam, track, frame = match.groups()
                    pid = int(pid)
                    cam = int(cam)
                    track = int(track)
                    frame = int(frame)

                    # 生成 Market-1501 格式的文件名
                    new_filename = f"{pid:04d}_c{cam}s{track}_{frame:06d}_00.jpg"

                    if 'bbox_train' in root:
                        train_files.append((pid, os.path.join(root, file), new_filename))
                    elif 'bbox_test' in root:
                        test_files.append((pid, os.path.join(root, file), new_filename))
                        if pid not in query_files:
                            query_files[pid] = []
                        query_files[pid].append((os.path.join(root, file), new_filename))

    # 按照每个 PID 抽取 25 张图片
    train_files_dict = {}
    for pid, src, dst in train_files:
        if pid not in train_files_dict:
            train_files_dict[pid] = []
        train_files_dict[pid].append((src, dst))

    test_files_dict = {}
    for pid, src, dst in test_files:
        if pid not in test_files_dict:
            test_files_dict[pid] = []
        test_files_dict[pid].append((src, dst))

    selected_train_files = []
    for pid, files in train_files_dict.items():
        if len(files) > num_images_per_id:
            selected_files = random.sample(files, num_images_per_id)
        else:
            selected_files = files
        selected_train_files.extend(selected_files)

    selected_test_files = []
    for pid, files in test_files_dict.items():
        if len(files) > num_images_per_id:
            selected_files = random.sample(files, num_images_per_id)
        else:
            selected_files = files
        selected_test_files.extend(selected_files)

    # 将选定的训练文件复制到 Market-1501 结构的 bounding_box_train 文件夹中
    train_dst_dir = os.path.join(market_path, 'bounding_box_train')
    for src, dst in tqdm(selected_train_files, desc="Copying training files", unit="file"):
        shutil.copy(src, os.path.join(train_dst_dir, dst))

    # 将选定的测试文件复制到 Market-1501 结构的 bounding_box_test 文件夹中
    test_dst_dir = os.path.join(market_path, 'bounding_box_test')
    for src, dst in tqdm(selected_test_files, desc="Copying test files", unit="file"):
        shutil.copy(src, os.path.join(test_dst_dir, dst))

    # 将每个 PID 的第一张测试文件复制到 query 文件夹中
    query_dst_dir = os.path.join(market_path, 'query')
    for pid in query_files:
        src, dst = query_files[pid][0]
        shutil.copy(src, os.path.join(query_dst_dir, dst))

    return train_dst_dir, test_dst_dir, query_dst_dir

def count_images_and_ids(directory):
    id_counts = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                pid = file.split('_')[0]
                if pid not in id_counts:
                    id_counts[pid] = 0
                id_counts[pid] += 1
    return id_counts

def print_summary(directory, label):
    id_counts = count_images_and_ids(directory)
    num_ids = len(id_counts)
    num_images = sum(id_counts.values())
    print(f"{label}: {num_images} images, {num_ids} IDs")

if __name__ == '__main__':
    mars_dataset_path = 'mars'  # 修改为 MARS 数据集的实际路径
    market1501_output_path = 'market1501'  # 修改为目标 Market-1501 数据集路径

    make_market_dir(market1501_output_path)
    train_dst_dir, test_dst_dir, query_dst_dir = extract_mars_to_market(mars_dataset_path, market1501_output_path)

    print_summary(train_dst_dir, "Training set")
    print_summary(test_dst_dir, "Test set")
    print_summary(query_dst_dir, "Query set")
