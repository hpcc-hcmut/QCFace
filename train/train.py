#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import warnings
import time
import pprint
from termcolor import cprint

import torch
import torch.nn as nn

import utils
from dataloader import dataloader_generation
from model.single import build_model


warnings.filterwarnings("ignore")

def str2list(v):
    if ',' in v:
        return v.split(',')
    else:
        return [v]

# parse the args
cprint('=> parse the args ...', 'green')
parser = argparse.ArgumentParser(description='Trainer for Face Recognition')
parser.add_argument('--arch', default='resnet100', type=str,
                    help='backbone architechture')
parser.add_argument('--loss_model', type=str, default='qcface')
parser.add_argument('--phase', default=None,
                    help='training phase')
parser.add_argument('--weight_path', type=str, default=None,
                    help='path of model weight for resuming')
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+',
                    help='gpu id')
parser.add_argument('--data_dirs', type=str2list,
                    help='<data_root_dir1>,<data_root_dir2>,...')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr-drop-epoch', default=[30, 60, 90], type=int, nargs='+',
                    help='The learning rate drop epoch')
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='The learning rate drop ratio')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--pth-save-fold', default='tmp', type=str,
                    help='The folder to save pths')
parser.add_argument('--pth-save-epoch', default=1, type=int,
                    help='The epoch to save pth')

# model parameters
parser.add_argument('--embed_dims', default=512, type=int,
                    help='embedded dimension')
parser.add_argument('--lambda_g', default=20, type=float,
                    help='the lambda for function g')
parser.add_argument('--vis_mag', default=1, type=int,
                    help='visualize the magnitude against cos')

args = parser.parse_args()


if isinstance(args.gpu_id, (tuple, list)) and len(args.gpu_id)==2:
    device = (torch.device(f'cuda:{args.gpu_id[0]}'), torch.device(f'cuda:{args.gpu_id[1]}'))
elif args.gpu_id is not None:
    device = torch.device(f'cuda:{args.gpu_id[0]}')
else:
    device = torch.device('cpu')


def main(args):
    # check the feasible of the lambda g
    cprint('=> torch version : {}'.format(torch.__version__), 'green')
    ngpus_per_node = torch.cuda.device_count()
    cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')

    main_worker(args)


def main_worker(args):
    global best_acc1

    cprint('=> building the dataloader ...', 'green')
    train_loader = dataloader_generation(args.data_dirs, args.batch_size, args.workers)

    args.num_classes = train_loader.dataset.num_classes
    cprint(f'=> number of identities are {args.num_classes}', 'green')

    cprint('=> modeling the network ...', 'green')
    model = build_model(args, device)
    if args.weight_path is not None:
        cprint(f'=> loading weight from {args.weight_path} ...', 'green')
        model.load_weight(args.weight_path)

    cprint('=> building the oprimizer ...', 'green')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    pprint.pprint(optimizer)

    cprint('=> building the criterion ...', 'green')
    criterion = nn.CrossEntropyLoss(reduction="mean")

    global iters
    iters = 0

    cprint('=> starting training engine ...', 'green')

    if args.loss_model == "qaface":
        model.backbone.train()
        model.mbackbone.eval()
    else:
        model.train()

    for epoch in range(args.start_epoch, args.epochs):

        global current_lr
        current_lr = utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, epoch, args)

        # save pth
        if epoch % args.pth_save_epoch == 0:
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict
            }, False,
                filename=os.path.join(
                args.pth_save_fold, '{}.pth'.format(
                    str(epoch+1).zfill(5))
            ))
            cprint(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.3f')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    learning_rate = utils.AverageMeter('LR', ':.7f')
    throughputs = utils.AverageMeter('ThroughPut', ':.2f')

    losses_id = utils.AverageMeter('L_ID', ':.3f')
    losses_mag = utils.AverageMeter('L_mag', ':.6f')
    progress_template = [batch_time, data_time, throughputs, 'images/s',
                         losses, losses_id, losses_mag, 
                         top1, top5, learning_rate]

    progress = utils.ProgressMeter(
        len(train_loader),
        progress_template,
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    # update lr
    learning_rate.update(current_lr)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        global iters
        iters += 1

        if not isinstance(device, (tuple, list)):
            input = input.cuda(device, non_blocking=True)
            target = target.cuda(device, non_blocking=True)
        else:
            target = target.cuda(torch.device(device[0]), non_blocking=True)

        # compute output
        output, norm, loss_g, one_hot = model(input, target)

        loss_id = criterion(output[1], target)
        loss = loss_id + args.lambda_g * loss_g

        # measure accuracy and record loss
        acc1, acc5 = utils.accuracy(output[0], target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        losses_id.update(loss_id.item(), input.size(0))
        losses_mag.update(args.lambda_g*loss_g.item() if isinstance(loss_g, torch.Tensor) else args.lambda_g*loss_g, input.size(0))

        # compute gradient and do solver step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        duration = time.time() - end
        batch_time.update(duration)
        end = time.time()
        throughputs.update(args.batch_size / duration)

        if i % args.print_freq == 0:
            progress.display(i)
            debug_info(norm)


def debug_info(x_norm):
    """
    visualize the magnitudes and magins during training.
    Note: modify the function if m(a) is not linear
    """
    mean_ = torch.mean(x_norm).detach().cpu().numpy()
    max_ = torch.max(x_norm).detach().cpu().numpy()
    min_ = torch.min(x_norm).detach().cpu().numpy()
    print('  [debug info]: x_norm mean: {:.2f} min: {:.2f} max: {:.2f}'
          .format(mean_, min_, max_))


if __name__ == '__main__':
    pprint.pprint(vars(args))
    main(args)
