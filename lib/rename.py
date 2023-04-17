import os
path = r"D:\BoneAge\CR\dataset\unlabelled"
fileList = os.listdir(path)
i = 20001
for file in fileList:
    old = os.path.join(path, file)
    j = str(i).zfill(5)
    new = os.path.join(path, j+'.png')
    i = i + 1
    os.rename(old, new)
    print(old, '======>', new)
