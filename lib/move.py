import os
import shutil

root = r"D:\BoneAge\DataSet\archive\Digital Hand Atlas\JPEGimages"
target = r"D:\BoneAge\CR\dataset\unlabeled"


second_root = os.listdir(root)
for second in second_root:
    image_second_root = os.path.join(root, second)
    image_names = os.listdir(image_second_root)
    print(image_second_root)
    for image_name in image_names:
        image_root = os.path.join(image_second_root, image_name)
        print(image_root)
        image = os.listdir(image_root)
        for img in image:
            img_path = os.path.join(image_root, img)
            print(img_path)
            shutil.copy(img_path, target)



