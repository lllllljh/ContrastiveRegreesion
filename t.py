import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

batch_size = 64

class MyDataset(Dataset):

    def __init__(self, data_dir, info_csv,transform=None):
        label_info = pd.read_csv(info_csv)
        image_file = os.listdir(data_dir)
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_file[index].split('.')[0]
        raw_label = self.label_info.loc[self.label_info['ID'].apply(lambda x: str(x) + '.png') == image_name]
        label = torch.tensor(raw_label['Boneage'])
        image_name = os.path.join(self.data_dir, self.image_file[index])
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.image_file)



#
# def get_mean_std_value(loader):
#     '''
#     求数据集的均值和标准差
#     :param loader:
#     :return:
#     '''
#     data_sum,data_squared_sum,num_batches = 0,0,0
#
#     for data,_ in loader:
#         # data: [batch_size,channels,height,width]
#         # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
#         data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
#         # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
#         data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
#         # 统计batch的数量
#         num_batches += 1
#     # 计算均值
#     mean = data_sum/num_batches
#     # 计算标准差
#     std = (data_squared_sum/num_batches - mean**2)**0.5
#     return mean,std
#
# mean,std = get_mean_std_value(train_loader)
# print('mean = {},std = {}'.format(mean,std))



def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    i = 0
    for X, _ in train_loader:
        print(i)
        i+=1
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((300, 400)),  # 将所有图像的大小调整为 224 x 224
        transforms.ToTensor()
    ])

    train_dataset = MyDataset(data_dir="D:/BoneAge/CR/dataset/all", info_csv="D:/BoneAge/CR/dataset/all.csv",
                              transform=transform)
    print(get_mean_and_std(train_dataset))
