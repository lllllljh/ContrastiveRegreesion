import os
import cv2

root = r"D:/BoneAge/DataSet/RHPE"
target = r"D:/BoneAge/DataSet/temp1"
imglist = os.listdir(root)
i = 20000
for img_name in imglist:
    img_path = os.path.join(root, img_name)
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    w = int(w * 0.7)
    img = img[0:h, 0:w]
    cv2.imwrite(os.path.join(target, str(i)+'.png'), img)
    i += 1

