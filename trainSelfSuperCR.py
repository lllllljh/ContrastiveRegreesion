import math
import os
import sys
import argparse

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
from network.SelfSuperCRlosses import SelfCRLoss
from network.Resnet import CR


class TwoCropTransform:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


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

        return images

    def __len__(self):
        return len(self.image_file)


def set_model(opt):
    model = CR()
    criterion = SelfCRLoss(temperature=opt.temp, base_temperature=opt.base_temp)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def set_data_loader(opt):
    normalize = transforms.Normalize(mean=eval(opt.mean), std=eval(opt.std))

    train_transform = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 400)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ])

    train_data_path = os.path.join(opt.dataset_path, 'unlabelledtrain')
    val_data_path = os.path.join(opt.dataset_path, 'unlabelledval')

    train_dataset = MyDataset(
        data_dir=train_data_path, transform=TwoCropTransform(train_transform)
    )
    train_loader = dataloader.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True
    )
    val_dataset = MyDataset(
        data_dir=val_data_path, transform=TwoCropTransform(val_transform)
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


def save_model(model, optimizer, opt, epoch, save_file):
    print('Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def train(train_loader, model, criterion, optimizer, epoch, opt):
    model.train()
    losses = AverageMeter()

    for i, (images) in enumerate(train_loader):
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        batch_size = int (images.shape[0] / 2)
        features = model(images)
        f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
        losses.update(loss.item(), batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss: {loss.val:.3f} loss_avg:({loss.avg:.3f})'.format(epoch, i + 1, len(train_loader), loss=losses))
            sys.stdout.flush()

    # for name, parms in model.named_parameters():
    #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
    #           ' -->grad_value:', torch.mean(parms.grad))

    return losses.avg



def validate(val_loader, model, criterion, epoch, opt):

    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (images) in enumerate(val_loader):
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            batch_size = int (images.shape[0] / 2)
            features = model(images)
            f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features)
            losses.update(loss.item(), batch_size)

            if (i + 1) % opt.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'val_loss: {loss.val:.3f} val_loss_avg:({loss.avg:.3f})'.format(epoch, i + 1, len(val_loader),
                                                                                      loss=losses))
                sys.stdout.flush()

    return losses.avg

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset')
    parser.add_argument('--save_path', type=str, default='./output')
    parser.add_argument('--mean', type=str, default='(0.14770237,0.14770237,0.14770237)')
    parser.add_argument('--std', type=str, default='(0.17363681,0.17363681,0.17363681)')
    parser.add_argument('--size', type=int, default=256)

    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--temp', type=float, default=3.0)
    parser.add_argument('--base_temp', type=float, default=3.0)

    opt = parser.parse_args()

    train_name = datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime("%Y%m%d_%H%M%S")
    train_name = train_name + "PreEncoder"
    train_dir = os.path.join(opt.save_path, train_name)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    opt.model_path = train_dir
    opt.tb_path = os.path.join(train_dir, '{}_tensorboard'.format(train_name))

    return opt


if __name__ == '__main__':

    opt = parser_opt()
    train_loader, val_loader = set_data_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    logger = tensorboard_logger.Logger(logdir=opt.tb_path, flush_secs=2)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        logger.log_value('train_loss', loss, epoch)

        loss = validate(val_loader, model, criterion, epoch, opt)
        logger.log_value('val_loss', loss, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.model_path, 'epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    save_file = os.path.join(opt.model_path, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
