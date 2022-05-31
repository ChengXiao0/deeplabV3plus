import argparse
import numpy as np
import torch
import time
import torch.nn as nn
from model.deeplab import DeepLab
from data_loarder import makeDataLoader
from torch.optim import SGD
from customer_utils.loss import SegmentationLosses
from customer_utils.lr_scheduler import LR_Scheduler
from customer_utils.metrics import Evaluator

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--lr', type=float, help='enter learn rate', default=0.1)
parser.add_argument('--epoch', type=int, help='epoch of model', default=80)
parser.add_argument('--batchsize', type=int, help='batchsize of dataloader', default=32)
parser.add_argument('--numclass', type=int, help='coco -> 21\npascal -> 21', default=21)
parser.add_argument('--backbone', type=str, help='deeplab\'s backbone', default='mobilenet')
parser.add_argument('--cuda', type=list, default='0', help='cuda ids')
args = parser.parse_args()


class trainer():
    def __init__(self, args):
        self.epoch = args.epoch
        self.batchsize = args.batchsize
        self.lr = args.lr
        self.backbone = args.backbone
        self.cuda_ids = []
        self.best_pred = 0.
        self.numclass = args.numclass
        self.net = DeepLab(backbone=self.backbone, num_classes=self.numclass)
        self.train_params = [{'params': self.net.get_1x_lr_params(), 'lr': args.lr},
                             {'params': self.net.get_10x_lr_params(), 'lr': args.lr * 10}]
        self.trainLoader, self.valLoader = makeDataLoader(batchsize=self.batchsize)
        self.criterion = SegmentationLosses(cuda=True).build_loss()
        self.lr_scheduler = LR_Scheduler(mode='poly', base_lr=self.lr, num_epochs=self.epoch,
                                         iters_per_epoch=len(self.trainLoader))
        self.evaluator = Evaluator(num_class=21)
        for ids in args.cuda:
            if ids != ',':
                self.cuda_ids.append(eval(ids))
        # self.net = nn.DataParallel(self.net, self.cuda_ids)
        self.net = self.net.cuda()
        self.opti = torch.optim.SGD(self.train_params, momentum=0.9)
        # self.opti = SGD(self.train_params, momentum=0.9)

    def training(self):
        train_loss = 0.
        self.net.train()
        for epoch_train in range(self.epoch):
            for i, sample in enumerate(self.trainLoader):
                image, target = sample['image'], sample['label']
                image, target = image.cuda(), target.cuda()
                self.lr_scheduler(self.opti, i, epoch_train, self.best_pred)
                output = self.net(image)
                self.opti.zero_grad()
                loss = self.criterion(output, target)
                loss.backward()
                self.opti.step()
                train_loss += loss
                print(f'in the train->epoch:{epoch_train}->batch:{i}->loss:{loss.item()}')
            self.validation(epoch_train)

    def validation(self, valEpoch):
        self.net.eval()
        self.evaluator.reset()
        test_loss = 0.0
        for i, sample in enumerate(self.valLoader):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.net(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        if mIoU > self.best_pred:
            self.best_pred = mIoU
            torch.save(self.net.state_dict(), f'ckp/resnet101/model{valEpoch}_{mIoU}.pth')


x = trainer(args)
x.training()

