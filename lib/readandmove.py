import os
import shutil

root = r"D:\BoneAge\DataSet"
info_path = r"D:\BoneAge\DataSet\MURA-v1.1\valid_image_paths.txt"
tagret_path = r"D:\BoneAge\CR\dataset\unlabeled3"

info = open(info_path, mode='r')

lines = info.readlines()

i = 30000
for item in lines:
    file_path = os.path.join(root, item)
    file_path = file_path.rstrip("\n")
    name = file_path.split('/')[5]
    shutil.copy(file_path, tagret_path)
    new = os.path.join(tagret_path, str(i)+'.png')
    old = os.path.join(tagret_path, name)
    i = i + 1
    os.rename(old, new)



