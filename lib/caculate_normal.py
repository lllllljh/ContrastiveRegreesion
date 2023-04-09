import os
from PIL import Image
import numpy as np
import tqdm

path = r"D:\BoneAge\DataSet\all"
img_channels = 3
img_names = os.listdir(path)
cumulative_mean = np.zeros(img_channels)
cumulative_std = np.zeros(img_channels)

for img_name in tqdm.tqdm(img_names, total=len(img_names)):
    img_path = os.path.join(path, img_name)
    img = np.array(Image.open(img_path)) / 255.
    print(img_path)
    for d in range(3):
        cumulative_mean[d] += img[:, :, d].mean()
        cumulative_std[d] += img[:, :, d].std()
    mean = cumulative_mean / len(img_names)
    std = cumulative_std / len(img_names)
    print(f"mean: {mean}")
    print(f"std: {std}")

