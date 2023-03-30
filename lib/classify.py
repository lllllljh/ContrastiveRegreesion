import os
import shutil
import pandas

root = "../dataset/"
csv_path = open("../dataset/dataset.csv")
data_path = "../dataset/dataset/"
list = pandas.read_csv(csv_path)

for i in range(1, 229):
    new_list = list[list["Boneage"] == i]
    group = new_list["ID"].tolist()
    for j in group:
        new_path = os.path.join(root, str(i))
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        shutil.move(os.path.join(data_path, str(j)+".png"), new_path)














