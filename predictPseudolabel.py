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
from network.Resnet import CR, RegressionWithoutLabel


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
    def __init__(self, data_dir, transform=None):

        image_file = os.listdir(data_dir)
        self.data_dir = data_dir
        self.image_file = image_file
        self.transform = transform

    def __getitem__(self, index):
        image_name = os.path.join(self.data_dir, self.image_file[index])
        images = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        return images, self.image_file[index]

    def __len__(self):
        return len(self.image_file)


def set_model(opt):
    model = CR()
    regression = RegressionWithoutLabel()
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
        transforms.Resize((300, 400)),
        transforms.ToTensor(),
        normalize
    ])

    test_data_path = os.path.join(opt.dataset_path, 'unlabel')


    test_dataset = MyDataset(
        data_dir=test_data_path, transform=test_transform
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

    with torch.no_grad():
        for i, (images, names) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            features = model(images)
            out = regression(features)

            print('{0}\t{1:.3f}\t'.format(names[0], out[0].item()))
            print('{0}\t{1:.3f}\t'.format(names[0], out[0].item()), file=log)


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--weight', type=str, default='./weight/last.pth')
    parser.add_argument('--mean', type=str, default='(0.115339115, 0.115339115, 0.115339115)')
    parser.add_argument('--std', type=str, default='(0.18438558, 0.18438558, 0.18438558)')
    parser.add_argument('--size', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--workers', type=int, default=4)

    opt = parser.parse_args()

    test_name = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    test_name = test_name + "Pseudo label"
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
    test(test_loader, model, regression, log)
    log.close()
