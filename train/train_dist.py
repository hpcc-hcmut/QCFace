#!/usr/bin/env python
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import warnings
import time
import random
import pprint
from termcolor import cprint

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchshard as ts
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.multiprocessing as mp

import utils
from dataloader import distdataloader_generation
from model.dist import build_dist_model
from loss import DistFaceLoss


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
parser.add_argument('--lambda_g', default=1.0, type=float,
                    help='the lambda for function g')

# parallel configuration
parser.add_argument('--node_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpus_per_node', default=2, type=int,
                    help='gpu count per node')
parser.add_argument('--num_nodes', default=1, type=int,
                    help='node count')
parser.add_argument('--master_addr', default='172.21.0.2', type=str,
                    help='IP address of master node')
parser.add_argument('--socket', default='eth0', type=str,
                    help='NCCL socket name')
parser.add_argument('--port', default='12355', type=str,
                    help='Master port')

parser.add_argument('--vis_mag', default=1, type=int,
                    help='visualize the magnitude against cos')

args = parser.parse_args()


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def main_worker(gpu, args):
    # Setup for distributed training
    ngpus_per_node = torch.cuda.device_count()
    args.gpu = gpu
    args.rank = args.nr * args.gpus + args.gpu
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    init_seeds(0+args.rank)  
    
    # logs
    if args.rank == 0:
        cprint('=> torch version : {}'.format(torch.__version__), 'green')
        cprint('=> ngpus : {}'.format(ngpus_per_node), 'green')
    
    # init torchshard
    ts.distributed.init_process_group(group_size=args.world_size)
    
    # training main
    global best_acc1

    if args.rank == 0:
        cprint('=> building the dataloader ...', 'green')
    train_loader = distdataloader_generation(args.data_dirs, args.batch_size, args.workers, args.rank, args.world_size)

    if args.rank == 0:
        cprint(f'=> number of identities are {train_loader.dataset.num_classes}', 'green')
        cprint('=> modeling the network ...', 'green')
    
    model, loss_softmax_calc, loss_g_calc = build_dist_model(args)

    if args.rank == 0:
        cprint('=> building the oprimizer ...', 'green')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay) 
    if args.rank == 0:
        pprint.pprint(optimizer)

    grad_scaler = GradScaler(enabled=False)

    if args.rank == 0:
        cprint('=> building the criterion ...', 'green')
    criterion = DistFaceLoss(args.loss_model, loss_softmax_calc, loss_g_calc)

    global iters
    iters = 0

    if args.rank == 0:
        cprint('=> starting training engine ...', 'green')
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        global current_lr
        current_lr = utils.adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        do_train(train_loader, model, criterion, optimizer, grad_scaler, epoch, args)

        # ts.collect_state_dict() needs to see all the process groups
        state_dict = model.state_dict()
        state_dict = ts.collect_state_dict(model, state_dict)

        # save pth
        if epoch % args.pth_save_epoch == 0 and args.rank == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
            }, False,
                filename=os.path.join(
                args.pth_save_fold, '{}.pth'.format(
                    str(epoch+1).zfill(5))
            ))
            cprint(' : save pth for epoch {}'.format(epoch + 1))


def do_train(train_loader, model, criterion, optimizer, grad_scaler, epoch, args):
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

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        with autocast(enabled=False):
            output, norms = model(input)

        # x_norm is not needed to be gathered, as feature x is in each rank
        target = ts.distributed.gather(target, dim=0)
        
        # loss
        with autocast(enabled=False):
            loss_id, loss_g, one_hot = criterion(output, target, norms)
            loss_g = args.lambda_g * loss_g
        loss = loss_id + loss_g

        # compute gradient and do solver step
        optimizer.zero_grad()
        # backward
        grad_scaler.scale(loss).backward()
        # update weights
        grad_scaler.step(optimizer)
        grad_scaler.update() 

        # update for memory module
        if model.loss_model=="qaface":
            model.step()

        # syn for logging
        torch.cuda.synchronize()

        # measure elapsed time
        if args.rank == 0:
            duration = time.time() - end
            end = time.time()
            batch_time.update(duration)
            throughputs.update(args.world_size * args.batch_size / duration)

        # measure accuracy and record loss
        output = ts.distributed.gather(output[0], dim=-1)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        losses_id.update(loss_id.item(), input.size(0))
        losses_mag.update(loss_g.item() if isinstance(loss_g, torch.Tensor) else loss_g, input.size(0))

        if i % args.print_freq == 0 and args.rank == 0:
            progress.display(i)
            debug_info(norms)


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
    args.gpus = args.gpus_per_node
    args.nodes = args.num_nodes
    args.nr = args.node_rank    
    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_ADDR'] = args.master_addr              
    os.environ['NCCL_SOCKET_IFNAME'] = args.socket 
    os.environ['MASTER_PORT'] = args.port

    pprint.pprint(vars(args))
    
    if (args.batch_size % args.world_size) != 0:
        print('batch size {} is not a multiplier of world size {}'.format(
            args.batch_size, args.world_size
        ))
        exit(1)
    args.batch_size = int(args.batch_size / args.world_size)
    
    mp.spawn(main_worker, nprocs=args.gpus, args=(args,)) 
