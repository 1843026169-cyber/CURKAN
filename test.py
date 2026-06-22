import argparse
import os
import time
import random
import numpy as np
import setproctitle
import logging
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
from torch.utils.data import DataLoader
from data.BraTS import BraTS
from predict import validate_softmax
from models.UNETR import UNETR
from curkan import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--user', default = 'wsl', type = str)
parser.add_argument('--root', default = './Datasets', type = str)
parser.add_argument('--valid_dir', default = 'EADC', type = str)
parser.add_argument('--valid_file', default = 'test.txt', type = str)
parser.add_argument('--output_dir', default = 'output', type = str)
parser.add_argument('--submission', default = 'submission', type = str)
parser.add_argument('--visual', default = 'visualization', type = str)
parser.add_argument('--experiment', default = 'UNETR', type = str)
parser.add_argument('--test_date', default = '2026-06-01', type = str)
parser.add_argument('--test_file', default = 'model_epoch_last.pth', type = str)
parser.add_argument('--use_TTA', default = False, type = bool)
parser.add_argument('--post_process', default = False, type = bool)
parser.add_argument('--save_format', default = 'nii', choices = ['npy', 'nii'], type = str)
parser.add_argument('--crop_H', default = 128, type = int)
parser.add_argument('--crop_W', default = 128, type = int)
parser.add_argument('--crop_D', default = 128, type = int)
parser.add_argument('--seed', default = 1000, type = int)
parser.add_argument('--model_name', default = 'UNETR', type = str)
parser.add_argument('--num_cls', default = 2, type = int)
parser.add_argument('--no_cuda', default = False, type = bool)
parser.add_argument('--gpu', default = '1', type = str)
parser.add_argument('--num_workers', default = 2, type = int)
parser.add_argument('--lora_dim', default = 2, type = int)
args = parser.parse_args()

def setup_logging():
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.test_date)
    log_file = log_dir + '.txt'
    os.makedirs(log_dir, exist_ok = True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    fh = logging.FileHandler(log_file, encoding = 'utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

logger = setup_logging()

def main():
    code_filename = os.path.basename(__file__)
    logging.info('-----------------------------------------------')
    logger.info(f"测试文件: {code_filename}")
    logging.info(f"数据集: {args.valid_dir}")
    logger.info(f"lora_dim: {args.lora_dim}")
    logger.info(f"test_date: {args.test_date}")

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

    if args.lora_dim > 0:
        model = convert_linear_layer_to_curkan(model, 'vit', args.lora_dim)
        only_optimize_lora_parameters(model.vit)
    model = model.cuda()

    load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                             'checkpoint', args.experiment + args.test_date, args.test_file)

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file, weights_only = False)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']

        print(f"Successfully load checkpoint {os.path.join(args.experiment + args.test_date, args.test_file)}")
    else:
        print('There is no resume file to load!')

    valid_list = os.path.join(args.root, args.valid_dir, args.valid_file)
    valid_root = os.path.join(args.root, args.valid_dir)
    valid_set = BraTS(valid_list, valid_root, mode = 'test')

    logger.info(f'Samples for valid : {len(valid_set)}')

    valid_loader = DataLoader(valid_set, batch_size = 1, shuffle = False, num_workers = args.num_workers,
                              pin_memory = True)
    total_samples = len(valid_loader)

    submission = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                              args.submission, args.experiment + args.test_date)
    visual = os.path.join(os.path.abspath(os.path.dirname(__file__)), args.output_dir,
                          args.visual, args.experiment + args.test_date)

    if not os.path.exists(submission):
        os.makedirs(submission)
    if not os.path.exists(visual):
        os.makedirs(visual)

    start_time = time.time()

    with torch.no_grad():
        for idx, data in enumerate(valid_loader):
            print(f"当前测试进度：{idx + 1}/{total_samples}", end = '\r')

        validate_softmax(valid_loader = valid_loader,
                         model = model,
                         load_file = load_file,
                         multimodel = False,
                         savepath = submission,
                         verbose = True,
                         visual = visual,
                         names = valid_set.names,
                         use_TTA = args.use_TTA,
                         save_format = args.save_format,
                         snapshot = False,
                         postprocess = False
                         )

        print()

    end_time = time.time()
    full_test_time = (end_time - start_time) / 60
    average_time = full_test_time / len(valid_set)
    logger.info(f'测试时间：{full_test_time:.2f} minutes!')


if __name__ == '__main__':
    setproctitle.setproctitle('{}: Testing!'.format(args.user))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    main()