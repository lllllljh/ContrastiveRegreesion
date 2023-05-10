import os
import argparse

import pandas as pd
import pytz
import torch
import torch.utils.data as dataloader
from PIL import Image

from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
from network.ResNet18 import ResNet18
from network.Regression import Regression


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, transform=None):
        label_info = pd.read_csv(info_csv)
        image_file = os.listdir(data_dir)
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_file[index].split('.')[0]
        raw_label = self.label_info.loc[self.label_info['ID'].astype(str) == image_name]
        labels = torch.tensor(raw_label['Boneage'].values, dtype=torch.float32)
        sexs = torch.tensor(raw_label['Male'].values, dtype=torch.float32)
        image_name = os.path.join(self.data_dir, self.image_file[index])
        images = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        return images, labels, sexs, self.image_file[index]

    def __len__(self):
        return len(self.image_file)


def set_model(opt):
    model = ResNet18()
    regression = Regression()
    ckpt = torch.load(opt.weight, map_location='cpu')
    model_state_dict = ckpt['model']
    model.load_state_dict(model_state_dict)
    regression_state_dict = ckpt['regression']
    regression.load_state_dict(regression_state_dict)

    if torch.cuda.is_available():
        model = model.cuda()
        regression = regression.cuda()
        cudnn.benchmark = True

    return model, regression


def set_data_loader(opt):
    normalize = transforms.Normalize(mean=eval(opt.mean), std=eval(opt.std))

    test_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        normalize
    ])

    test_data_path = os.path.join(opt.dataset_path, 'test')
    test_info_path = os.path.join(opt.dataset_path, 'boneage_test.csv')

    test_dataset = MyDataset(
        data_dir=test_data_path, info_csv=test_info_path, transform=test_transform
    )
    test_loader = dataloader.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True
    )

    return test_loader


def accuracy(output, labels):
    mae = torch.abs(output - labels)
    return mae.mean()


def test(test_loader, model, regression, log):
    model.eval()
    regression.eval()
    acces = AverageMeter()
    with torch.no_grad():
        for i, (images, labels, sexes, names) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                sexes = sexes.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            features = model(images)
            out = regression(features, sexes)
            acc = abs(labels[0].item() - out[0].item())
            acces.update(acc, 1)
            print('Name:{0}\tTrueBoneAge:{1:.3f}\tPredictBoneAge:{2:.3f}\tAbsoluteError(month):{3:.3f}'.format(
                names[0],
                labels[0].item(),
                out[0].item(), acc))
            print('Name:{0}\tTrueBoneAge:{1:.3f}\tPredictBoneAge:{2:.3f}\tAbsoluteError(month):{3:.3f}'.format(
                names[0],
                labels[0].item(),
                out[0].item(), acc), file=log)

    return acces.sum


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--weight', type=str, default='./weight/SemiPredictorBest.pth')
    parser.add_argument('--mean', type=str, default='(0.115339115, 0.115339115, 0.115339115)')
    parser.add_argument('--std', type=str, default='(0.18438558, 0.18438558, 0.18438558)')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--workers', type=int, default=8)

    opt = parser.parse_args()

    test_name = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    test_name = test_name + "Test"
    test_dir = os.path.join(opt.save_path, test_name)
    log_dir = os.path.join(test_dir, 'log.txt')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        log = open(log_dir, mode='a+')
        log.close()
    opt.log_path = log_dir

    return opt


if __name__ == '__main__':
    opt = parser_opt()
    test_loader = set_data_loader(opt)
    model, regression = set_model(opt)
    log = open(opt.log_path, mode='a+')
    sum = test(test_loader, model, regression, log)
    print('Mean of Absolute Error:{0}'.format(sum * 1.0 / opt.test_num))
    print('Mean of Absolute Error:{0}'.format(sum * 1.0 / opt.test_num), file=log)
    log.close()
