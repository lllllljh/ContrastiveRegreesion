import os
path = r"D:\BoneAge\CR\dataset\unlabeled3"
fileList = os.listdir(path)
i = 34000
for file in fileList:
    old = os.path.join(path, file)
    new = os.path.join(path, str(i)+'.png')
    i = i + 1
    os.rename(old, new)
    print(old, '======>', new)
