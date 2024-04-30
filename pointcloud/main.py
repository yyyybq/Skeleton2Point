"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
#from torchlight import DictAction
import sys
import yaml
import gc
import shutil

import inspect
import torch.nn as nn
import pickle
from torchlight import DictAction
device_ids = [0,1,2,3]


import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
torch.backends.cudnn.enable =True
#导入类
class BTwins(nn.Module):
    def __init__(self, hidden_size1=102400, hidden_size2=102400, lambd=2e-4, pj_size=256):
        super().__init__()
        self.projector1 = nn.Sequential(
            nn.Linear(hidden_size1, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
        )
        self.projector2 = nn.Sequential(
            nn.Linear(hidden_size2, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
            nn.BatchNorm1d(pj_size),
            nn.ReLU(True),
            nn.Linear(pj_size, pj_size, bias=False),
        )

        self.bn = nn.BatchNorm1d(pj_size, affine=False)
        self.lambd = lambd

    def forward(self, feat1, feat2):
        #print(feat1.shape)
        feat1 = self.projector1(feat1)
        feat2 = self.projector2(feat2.permute(0,2,1))
        feat1_norm = self.bn(feat1)
        feat2_norm = self.bn(feat2)

        N, D = feat1_norm.shape
        c = (feat1_norm.T @ feat2_norm).div_(N)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        BTloss = on_diag + self.lambd * off_diag

        return BTloss 

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def parse_args():
    """Parameters"""
    
    parser = argparse.ArgumentParser('training')
    parser.add_argument(
        '--config',
        default='/data/yinbaiqiao/pointcloud/config/nturgbd-cross-subject/default.yaml',
        help='path to the configuration file')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default=' PointConT', help='model name [default: pointnet_cls]')

    parser.add_argument('--model2', default='ske_mixf', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=60, type=int, help='default value for classes of NTU60')#在这里改
    parser.add_argument('--epoch', default=250, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=3200, help='Point Number')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    # feeder
    parser.add_argument(
        '--feeder', default='feeders.feeder_ntu.Feeder', help='data loader will be used')
    parser.add_argument(
        '--train-feeder-args',
        #action=DictAction,
        #default=dict(),
        default={'data_path': '/data/yinbaiqiao/skeNTU60_CS.npz','split': 'train','window_size': 64,'p_interval': [0.5, 1]},
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        #action=DictAction,
        #default=dict(),
        default={'data_path': '/data/yinbaiqiao/skeNTU60_CS.npz','split': 'test','window_size': 64, 'p_interval': [0.95]},
        help='the arguments of data loader for test')
    return parser.parse_args(),parser


def main():
    #args = parse_args()
    _,parser = parse_args()

    # load arg form config file
    p = parser.parse_args()

    with open(p.config, 'r') as f:
        default_arg = yaml.load(f,Loader=yaml.FullLoader)
    #key = vars(p).keys()

    parser.set_defaults(**default_arg)

    args = parser.parse_args()

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'#Yes
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)#No
    else:
        device = 'cpu'
    #device=torch.device("cuda:2" )
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str #Yes
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)
    #args.checkpoint ='/data/yinbaiqiao/pointcloud/checkpoints/dela'
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    #这里不知道要干嘛
    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model](num_classes=args.num_classes)
    model2 = models.__dict__[args.model2]#(num_classes=args.num_classes)
    criterion = cal_loss #cross entropy loss, apply label smoothing if needed
    # net = net.to(device)
    
    # if device == 'cuda': #yes
    #     net = torch.nn.DataParallel(net,device_ids=device_ids)
    #     cudnn.benchmark = True
    output_device = device_ids[0] if type(device_ids) is list else device_ids
    load_device = device_ids[1]

    best_test_acc = 0.  # best test accuracy 
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    print('Load SkeMixformer..')
    #model2 = import_class(args.model2)
    #shutil.copy2(inspect.getfile(Model), arg.work_dir)
    print(model2)
    model2 = model2(**args.model_args)
    weights = torch.load("/data/yinbaiqiao/Skeleton-MixFormer-main/weight/NTU_60_Csub_J8.pt")

    weights = OrderedDict([[k.split('module.')[-1], v.cuda(load_device)] for k, v in weights.items()])

    keys = list(weights.keys())
    for w in args.ignore_weights:
        for key in keys:
            if w in key:
                if weights.pop(key, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(key))

    try:
        model2.load_state_dict(weights)
    except:
        state = model2.state_dict()
        diff = list(set(state.keys()).difference(set(weights.keys())))
        print('Can not find these weights:')
        for d in diff:
            print('  ' + d)
        state.update(weights)
        model2.load_state_dict(state)



    net = net.cuda(output_device)
    #self.model2 = self.model2.cuda(self.output_device)
    model2 = model2.cuda(load_device)
    #criterion = criterion.cuda(output_device)

    printf('==> Preparing data..')

    Feeder = import_class(args.feeder)
    train_loader = DataLoader(Feeder(**args.train_feeder_args), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Feeder(**args.test_feeder_args), num_workers=args.workers,
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    btwins_head = BTwins().cuda()
    optimizer = torch.optim.SGD([{'params':net.parameters()},{'params':btwins_head.parameters()}], lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:#NO
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net,model2, train_loader, optimizer, criterion, output_device, load_device)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(net, test_loader, criterion, output_device,epoch)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        if epoch>200:
            save_model(
                net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
                best_test_acc=best_test_acc,  # best test accuracy
                best_train_acc=best_train_acc,
                best_test_acc_avg=best_test_acc_avg,
                best_train_acc_avg=best_train_acc_avg,
                best_test_loss=best_test_loss,
                best_train_loss=best_train_loss,
                optimizer=optimizer.state_dict()
            )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
        
        gc.collect()
        torch.cuda.empty_cache()
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


def train(net,pre_net, trainloader, optimizer, criterion, output_device, load_device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.cuda(load_device), label.cuda(output_device).squeeze()
        out1, out2 = pre_net(data)
        #data = data.permute(0, 2, 1)  # so, the input data shape is [batch, 3, 1024]
        optimizer.zero_grad()
        logits,x2c,x3c = net(data.cuda(output_device),None, None)
        # print(x3c.size())# 16,64,1024
        # print(out2.size())#16,320,400

        losscls = criterion(logits, label).cuda(output_device)
        #btwins2 =  BTwins(hidden_size=256000).cuda(output_device)
        btwins3 =  BTwins(hidden_size1=64,hidden_size2=320).cuda(output_device)
        #cl_loss1 = btwins1(x1c.cuda(self.output_device),xb1.cuda(self.output_device))
        #cl_loss2 = btwins2(x2c.cuda(output_device),out1.cuda(output_device))
        cl_loss3 = btwins3(x3c.cuda(output_device),out2.cuda(output_device))
        #cl_loss1 = torch.mean(cl_loss1)
        #cl_loss2 = torch.mean(cl_loss2)
        cl_loss3 = torch.mean(cl_loss3)
        loss = losscls*0.9 + cl_loss3*0.1#cl_loss2*0.05 + cl_loss3*0.05
        
        loss.backward()
        #logits.shape=torch.Size([64, 15])
        #label.shape=torch.Size([64])
        
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device,epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    score_frag = []
    save_score=True
    time_cost = datetime.datetime.now()
    with torch.no_grad():#源码已加
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.cuda(output_device), label.cuda(output_device).squeeze()
            #data = data.permute(0, 2, 1)
            logits = net(data,None,None)
            loss = criterion(logits, label)
            score_frag.append(logits.data.cpu().numpy())
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    score = np.concatenate(score_frag)
    score_dict = dict(
                zip(testloader.dataset.sample_name, score))
    if save_score and epoch>180:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        '/data/yinbaiqiao/pointcloud', epoch + 1, 'test'), 'wb') as f:
                    pickle.dump(score_dict, f)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()
