import argparse
import os
import random
import logging
import numpy as np
import time
import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist
from models import criterionsWT
from models.criterions import *
from data.BraTS import BraTS
from torch.utils.data import DataLoader
from utils.tools import all_reduce_tensor
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from models.UNETR import UNETR
from curkan import *
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default = 'wangsl', type = str)
parser.add_argument('--experiment', default = 'UNETR', type = str)
parser.add_argument('--date', default = local_time.split(' ')[0], type = str)
parser.add_argument('--description',
                    default = 'UNETR,'
                              'training on train.txt!',
                    type = str)
# DataSet Information
parser.add_argument('--root', default = './Datasets', type = str)
parser.add_argument('--train_dir', default = 'EADC', type = str)
parser.add_argument('--val_dir', default = 'EADC', type = str)
parser.add_argument('--mode', default = 'train', type = str)
parser.add_argument('--train_file', default = 'train.txt', type = str)
parser.add_argument('--val_file', default = 'valid.txt', type = str)
parser.add_argument('--dataset', default = 'EADC', type = str)
parser.add_argument('--model_name', default = 'UNETR', type = str)

parser.add_argument('--lr', default = 0.001, type = float)
parser.add_argument('--weight_decay', default = 2e-5, type = float)
parser.add_argument('--amsgrad', default = True, type = bool)
parser.add_argument('--criterion', default = 'softmax_dice2', type = str)
parser.add_argument('--num_cls', default = 1, type = int)
parser.add_argument('--seed', default = 1000, type = int)
parser.add_argument('--no_cuda', default = False, type = bool)
parser.add_argument('--gpu', default = '1', type = str)
parser.add_argument('--num_workers', default = 2, type = int)
parser.add_argument('--batch_size', default = 2, type = int)
parser.add_argument('--start_epoch', default = 0, type = int)
parser.add_argument('--end_epoch', default = 1000, type = int)
parser.add_argument('--val_epoch', default = 100, type = int)
parser.add_argument('--save_freq', default = 500, type = int)
parser.add_argument('--resume', default = '', type = str)
parser.add_argument('--load', default = True, type = bool)

parser.add_argument('--lora_dim', default = 2, type = int)
args = parser.parse_args()

def freeze_parameters(model, param_names_to_freeze):
    for name, param in model.named_parameters():
        if name in param_names_to_freeze:
            param.requires_grad = False


param_names_to_freeze = [
    "encoder1.layer.conv1.conv.weight",
    "encoder1.layer.conv2.conv.weight",
    "encoder1.layer.conv3.conv.weight",
    "encoder2.transp_conv_init.conv.weight",
    "encoder2.blocks.0.0.conv.weight",
    "encoder2.blocks.0.1.conv1.conv.weight",
    "encoder2.blocks.0.1.conv2.conv.weight",
    "encoder2.blocks.1.0.conv.weight",
    "encoder2.blocks.1.1.conv1.conv.weight",
    "encoder2.blocks.1.1.conv2.conv.weight",
    "encoder3.transp_conv_init.conv.weight",
    "encoder3.blocks.0.0.conv.weight",
    "encoder3.blocks.0.1.conv1.conv.weight",
    "encoder3.blocks.0.1.conv2.conv.weight",
    "encoder4.transp_conv_init.conv.weight"
]


def main_worker():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    logging.info('-----------------------------------------------')
    logging.info('------------------training!!!!!----------------')

    code_filename = os.path.basename(__file__)
    logging.info(f"训练文件: {code_filename}")
    logging.info(f"数据集: {args.train_dir}")
    logging.info(f"lora_dim: {args.lora_dim}")
    logging.info(f"batch_size: {args.batch_size}")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model = UNETR(
        in_channels = 1,
        out_channels = 2,
        img_size = (128, 128, 128),
        feature_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_heads = 12,
        proj_type = "conv",
        norm_name = "instance",
        res_block = True,
        dropout_rate = 0.1
    )

    criterionWT = getattr(criterionsWT, args.criterion)

    checkpoint_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'checkpoint', args.experiment + args.date)
    os.makedirs(checkpoint_dir, exist_ok = True)

    results_file = os.path.join(checkpoint_dir, 'ckr_r64_LiTS.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            train_results = json.load(f)
    else:
        train_results = {
            'epoch': [],
            'loss': [],
            'dice_WT': [],
            'gpu_allocated': [],
            'gpu_reserved': [],
            'epoch_time': [],
            'gpu_peak_memory': []  # 新增记录峰值显存
        }

    resume = './checkpoint/UNETR2024-05-23/model_epoch_last.pth'
    if os.path.isfile(resume):
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location = lambda storage, loc: storage, weights_only = False)
        checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('re-training!!!')

    if args.lora_dim > 0:
        model = convert_linear_layer_to_curkan(model, 'vit', args.lora_dim)
        only_optimize_lora_parameters(model.vit)
        freeze_parameters(model, param_names_to_freeze)

    parameter_groups = get_optimizer_grouped_parameters(model, args.weight_decay)
    optimizer = torch.optim.Adam(parameter_groups, lr = args.lr)
    model.cuda()

    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    val_list = os.path.join(args.root, args.val_dir, args.val_file)
    val_root = os.path.join(args.root, args.val_dir)

    train_set = BraTS(train_list, train_root, args.mode)
    val_set = BraTS(val_list, val_root, args.mode)

    print('Samples for train = {}'.format(len(train_set)))
    logging.info(f"Samples for train: {len(train_set)}")
    print('Samples for val = {}'.format(len(val_set)))

    train_loader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True,
                              drop_last = True, num_workers = args.num_workers, pin_memory = True)
    val_loader = DataLoader(dataset = val_set, batch_size = args.batch_size, shuffle = True,
                            drop_last = True, num_workers = args.num_workers, pin_memory = True)

    start_time = time.time()
    torch.set_grad_enabled(True)
    epoch_iter_counts = []

    lambda_reg = 1e-5
    for epoch in range(args.start_epoch, args.end_epoch):
        setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
        start_epoch = time.time()
        print(f"当前训练Epoch：{epoch + 1}/{args.end_epoch}", end = '\r')

        model.train()
        for i, data in enumerate(tqdm(train_loader, disable = True)):
            adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
            x, target = data
            x = x.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)

            output = model(x)
            loss, loss_0, loss_1 = criterionWT(output, target)
            reduce_loss = loss.item()
            reduce_lossWT = loss_1.item()

            kan_reg_loss = 0.0
            for module in model.modules():
                if isinstance(module, TaylorKANLinear):
                    kan_reg_loss += module.regularization_loss()

            total_loss = loss + lambda_reg * kan_reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()
        current_epoch_seq = epoch - args.start_epoch + 1
        if current_epoch_seq <= 10:
            epoch_iter_counts.append(len(train_loader))
            if current_epoch_seq == 10:
                avg_iter_num = sum(epoch_iter_counts) / len(epoch_iter_counts)
                print(f"前10个Epoch的平均迭代次数: {avg_iter_num:.2f}")

        model.eval()
        dice_WT = 0.0
        val_count = 0
        if epoch % args.val_epoch == 0:
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    x, target = data
                    x = x.cuda(non_blocking = True)
                    target = target.cuda(non_blocking = True)
                    output = model(x)
                    dice_WT += Dice(output[:, 1, ...], (target > 0).float()).item()
                    val_count += 1
                dice_WT /= max(val_count, 1)

        end_epoch = time.time()
        epoch_duration = end_epoch - start_epoch
        torch.cuda.synchronize()
        gpu_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        gpu_allocated = torch.cuda.memory_allocated() / 1024 ** 2
        gpu_reserved = torch.cuda.memory_reserved() / 1024 ** 2

        train_results['epoch'].append(epoch)
        train_results['loss'].append(reduce_loss)
        train_results['dice_WT'].append(dice_WT)
        train_results['gpu_allocated'].append(gpu_allocated)
        train_results['gpu_reserved'].append(gpu_reserved)
        train_results['epoch_time'].append(epoch_duration)
        train_results['gpu_peak_memory'].append(gpu_peak)

        with open(results_file, 'w') as f:
            json.dump(train_results, f)

        if (epoch + 1) % int(args.save_freq) == 0 \
                or (epoch + 1) in [args.end_epoch - 1, args.end_epoch - 2, args.end_epoch - 3]:
            file_name = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            }, file_name)

    final_name = os.path.join(checkpoint_dir, 'model_epoch_last.pth')
    torch.save({
        'epoch': args.end_epoch,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict(),
    }, final_name)

    total_time = (time.time() - start_time) / 3600
    logging.info('训练时间： {:.2f} hours'.format(total_time))
    logging.info('-------------------训练结束--------------------')


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power = 0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def log_args(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()