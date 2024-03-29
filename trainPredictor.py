import math
import os
import sys
import argparse

import pandas as pd
import pytz
import torch
import torch.utils.data as dataloader
import tensorboard_logger
from PIL import Image

from torch import optim
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
from network.Regression import Regression
from network.ResNet18 import ResNet18


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

        return images, labels, sexs

    def __len__(self):
        return len(self.image_file)


def set_model(opt):

    model = ResNet18()
    ckpt = torch.load(opt.weight, map_location='cpu')
    model_state_dict = ckpt['model']
    model.load_state_dict(model_state_dict)
    regression = Regression()
    criterion = torch.nn.MSELoss()


    if torch.cuda.is_available():
        model = model.cuda()
        regression = regression.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, regression, criterion


def set_data_loader(opt):
    normalize = transforms.Normalize(mean=eval(opt.mean), std=eval(opt.std))

    train_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor(),
        normalize
    ])

    train_data_path = os.path.join(opt.dataset_path, 'train')
    train_info_path = os.path.join(opt.dataset_path, 'boneage_train.csv')
    val_data_path = os.path.join(opt.dataset_path, 'val')
    val_info_path = os.path.join(opt.dataset_path, 'boneage_val.csv')

    train_dataset = MyDataset(
        data_dir=train_data_path, info_csv=train_info_path, transform=train_transform
    )
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True
    )
    val_dataset = MyDataset(
        data_dir=val_data_path, info_csv=val_info_path, transform=val_transform
    )
    val_loader = dataloader.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True
    )

    return train_loader, val_loader


def adjust_learning_rate(opt, optimizer, epoch):
    lr = opt.learning_rate
    eta_min = lr * (opt.lr_decay_rate ** 3)
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / opt.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, regression, optimizer, opt, epoch, save_file):
    print('Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'regression': regression.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def accuracy(output, labels):
    with torch.no_grad():
        mae = torch.abs(output - labels)
        return mae.mean()


def train(train_loader, model, regression, criterion, optimizer, epoch, opt):

    model.eval()
    regression.train()
    losses = AverageMeter()
    acces = AverageMeter()

    for i, (images, labels, sexes) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            sexes = sexes.cuda(non_blocking=True)
        batch_size = labels.shape[0]
        with torch.no_grad():
            features = model(images)
        out = regression(features.detach(), sexes)
        loss = criterion(out, labels)
        losses.update(loss.item(), batch_size)
        acces.update(accuracy(out, labels), batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss: {loss.val:.3f} \tloss_avg:({loss.avg:.3f}) \t'
                  'acc: {acc.val:.3f} \tacc_avg:({acc.avg:.3f}) \t'.format(epoch, i + 1, len(train_loader), loss=losses,
                                                                         acc=acces))
            sys.stdout.flush()


    return losses.avg, acces.avg


def validate(val_loader, model, regression, criterion, epoch, opt):

    model.eval()
    regression.eval()
    losses = AverageMeter()
    acces = AverageMeter()

    with torch.no_grad():
        for i, (images, labels, sexes) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                sexes = sexes.cuda(non_blocking=True)
            batch_size = labels.shape[0]
            features = model(images)
            out = regression(features.detach(), sexes)
            loss = criterion(out, labels)
            losses.update(loss.item(), batch_size)
            acces.update(accuracy(out, labels), batch_size)

            if (i + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'val_loss: {loss.val:.3f} \tval_loss_avg:({loss.avg:.3f}) \t'
                      'val_acc: {acc.val:.3f} \tval_acc_avg:({acc.avg:.3f}) \t'.format(epoch, i + 1, len(val_loader),
                                                                              loss=losses, acc=acces))
                sys.stdout.flush()

    return losses.avg, acces.avg

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--weight', type=str, default='./weight/EncoderBest.pth')
    parser.add_argument('--mean', type=str, default='(0.115339115, 0.115339115, 0.115339115)')
    parser.add_argument('--std', type=str, default='(0.18438558, 0.18438558, 0.18438558)')
    parser.add_argument('--size', type=int, default=400)

    parser.add_argument('--print_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)

    opt = parser.parse_args()

    train_name = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    train_name = train_name + "SupPredictor"
    train_dir = os.path.join(opt.save_path, train_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    opt.model_path = train_dir
    opt.tb_path = os.path.join(train_dir, '{}_tensorboard'.format(train_name))

    return opt


if __name__ == '__main__':

    opt = parser_opt()
    train_loader, val_loader = set_data_loader(opt)
    model, regression, criterion = set_model(opt)
    optimizer = set_optimizer(opt, regression)
    logger = tensorboard_logger.Logger(logdir=opt.tb_path, flush_secs=2)

    min_acc = 50
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss, acc = train(train_loader, model, regression, criterion, optimizer, epoch, opt)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', acc, epoch)

        torch.cuda.empty_cache()

        loss, acc = validate(val_loader, model, regression, criterion, epoch, opt)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', acc, epoch)

        if min_acc > acc:
            min_acc = acc
            save_file = os.path.join(opt.model_path, 'best.pth'.format(epoch=epoch))
            save_model(model, regression, optimizer, opt, epoch, save_file)

    save_file = os.path.join(opt.model_path, 'last.pth')
    save_model(model, regression, optimizer, opt, opt.epochs, save_file)
