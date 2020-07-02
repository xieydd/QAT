import datetime
import os
import time
import sys
import copy
import numpy as np
import torch
import torch.utils.data
from torch import nn
# import torchvision
# from torchvision import transforms
import torch.quantization
import utils_train
from data.wider_face import TrainDataset, ValDataset, detection_collate,collate_eval
from data.data_argument import preproc,preproc_eval
from models.net_slim import Slim
from models.qat_slim import slim
from layers.functions.prior_box import PriorBox
from layers.modules.multibox_loss import MultiBoxLoss
from data.config import cfg_slim
from utils.eval_widerface import evaluate_widerface
try:
    from apex import amp
except ImportError:
    amp = None

def print_model_info(model,size):
    from torchprofile import profile_macs
    x = torch.randn(1,3,size,size)
    macs = profile_macs(model,x)
    params = np.sum(np.prod(v.size())
                    for name, v in model.named_parameters() if "auxiliary" not in name)

    print('{:<30}  {:<8}{:8}'.format(
        'Computational complexity: ', macs / 1e6, "M MACs"))
    print('{:<30}  {:<8}{:8}'.format(
        'Number of parameters: ', params / 1e6, " M"))

def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, priors,apex=False):
    model.train()
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils_train.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter(
        'img/s', utils_train.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        target = move_to(target, device)
        image = image.to(device)
        output = model(image)
        loss_l, loss_c,loss_landm = criterion(output,priors,target)
        loss = 2.0*loss_l + loss_c + loss_landm
        optimizer.zero_grad()
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        batch_size = len(target)
        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['losses'].update(loss.item(), n=batch_size)
        metric_logger.meters['box losses'].update(loss_l.item(), n=batch_size)
        metric_logger.meters['class losses'].update(loss_c.item(), n=batch_size)
        metric_logger.meters['landmark losses'].update(loss_landm.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size /
                                             (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, priors,print_freq=100):
    model.eval()
    metric_logger = utils_train.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            target = move_to(target, device)
            image = image.to(device)
            recall, precision = evaluate_widerface(image, target, model)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.meters['recall'].update(recall, n=batch_size)
            metric_logger.meters['precision'].update(precision, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' * Recall {recall.global_avg:.3f} Precision {precision.global_avg:.3f}'
          .format(recall=metric_logger.recall, precision=metric_logger.precision))
    return metric_logger.recall.global_avg


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision",
                              "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, cache_dataset, distributed,portion,debug=False):
    # Data loading code
    print("Loading data")
    normalize = (104, 117, 123)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = TrainDataset(traindir, preproc(300, normalize))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils_train.mkdir(os.path.dirname(cache_path))
            utils_train.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = ValDataset(traindir, preproc_eval(300, normalize))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils_train.mkdir(os.path.dirname(cache_path))
            utils_train.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    elif debug:
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(portion*num_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        num_test = len(dataset_test)
        indices_test = list(range(num_test))
        split_test = int(np.floor(portion*num_test))
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_test[:split_test])
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):

    if args.output_dir:
        utils_train.mkdir(args.output_dir)

    utils_train.init_distributed_mode(args)
    print(args)

    if args.post_training_quantize and args.distributed:
        raise RuntimeError("Post training quantization example should not be performed "
                           "on distributed mode")

    # Set backend engine to ensure that quantized model runs on the correct kernels
    if args.backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            "Quantized backend not supported: " + str(args.backend))
    torch.backends.quantized.engine = args.backend

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")
    train_dir = os.path.join(args.data_path, 'train/label.txt')
    val_dir = os.path.join(args.data_path, 'val/label.txt')

    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                   args.cache_dataset, args.distributed,args.portion,debug=args.debug)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True, collate_fn=detection_collate)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.eval_batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True, collate_fn=collate_eval)

    print("Creating model", args.model)
    cfg = None
    model = None
    qat = False
    if args.model == "slim":
        cfg = cfg_slim
        model = Slim(cfg=cfg)
    elif args.model == "slim-qat":
        cfg = cfg_slim
        #net = slim(args.test_only)
        qat = True
        model = slim(False)
    else:
        print("Don't support network!")
        exit(0)
    # when training quantized models, we always start from a pre-trained fp32 reference model
    model.to(device)
    print_model_info(net, 320)

    if not (args.test_only or args.post_training_quantize):
        if qat:
            model.fuse_model()
            model.qconfig = torch.quantization.get_default_qat_qconfig(
                args.backend)
            torch.quantization.prepare_qat(model, inplace=True)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                step_size=args.lr_step_size,
        #                                                gamma=args.lr_gamma)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[cfg['decay1'],cfg['decay2']],
                                                            gamma=args.lr_gamma)

    img_dim = cfg['image_size']
    num_classes = 2
    criterion = MultiBoxLoss(num_classes, 0.35,True,0,True,7,0.35,False)
    priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # TODO Office Bug, can`t load optimizer for gpu, will be fixed in pytorch 1.6 https://discuss.pytorch.org/t/bug-in-quantization-example/85492
    # If use resume model, set a new optimizer or don`t load optimizer and set model.to(device)

    if args.post_training_quantize:
        # perform calibration on a subset of the training dataset
        # for that, create a subset of the training dataset
        ds = torch.utils.data.Subset(
            dataset,
            indices=list(range(args.batch_size * args.num_calibration_batches)))
        data_loader_calibration = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
            pin_memory=True)
        model.eval()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig(args.backend)
        torch.quantization.prepare(model, inplace=True)
        # Calibrate first
        print("Calibrating")
        evaluate(model, criterion, data_loader_calibration,
                 device=device, print_freq=1, priors=priors)
        torch.quantization.convert(model, inplace=True)
        if args.output_dir:
            print('Saving quantized model')
            if utils_train.is_main_process():
                torch.save(model.state_dict(), os.path.join(args.output_dir,
                                                            'quantized_post_train_model.pth'))
        print("Evaluating post-training quantized model")
        evaluate(model, criterion, data_loader_test,
                 device=device, priors=priors)
        return

    if args.test_only:
        evaluate(model, criterion, data_loader_test,
                 device=device, priors=priors)
        return

    if qat:
        model.apply(torch.quantization.enable_observer)
        model.apply(torch.quantization.enable_fake_quant)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        print('Starting training for epoch', epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch,
                        args.print_freq, priors)
        lr_scheduler.step()
        with torch.no_grad():
            if qat:

                # Raghuraman Krishnamoorthi. Quantizing deep convolutional networks for efficient inference: A whitepaper.CoRR, abs/1806.08342, 2018.
                # Freeze BN param for avoid vanishing gradient 
                if epoch >= args.num_observer_update_epochs:
                    print('Disabling observer for subseq epochs, epoch = ', epoch)
                    model.apply(torch.quantization.disable_observer)
                if epoch >= args.num_batch_norm_update_epochs:
                    print('Freezing BN for subseq epochs, epoch = ', epoch)
                    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

                print('Evaluate QAT model')
            else:
                print('Evaluate  model')
            if epoch % args.eval_freq == 0:
                evaluate(model, criterion, data_loader_test,
                            device=device, priors=priors)

            
            #FIXME TODO: _cat for backend and evaluate only support cpu
            quantized_eval_model = copy.deepcopy(model_without_ddp)
            quantized_eval_model.eval()
            quantized_eval_model.to(torch.device('cpu'))
            if qat:
                torch.quantization.convert(quantized_eval_model, inplace=True)

                # print('Evaluate Quantized model')
                # evaluate(quantized_eval_model, criterion, data_loader_test,
                #          device=torch.device('cpu'), priors=priors)

        model.train()

        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'eval_model': quantized_eval_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            utils_train.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils_train.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))
        print('Saving models after epoch ', epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Training')

    parser.add_argument('--data-path',
                        default='./data/widerface/',
                        help='dataset')
    parser.add_argument('--model',
                        default='slim-qat',
                        help='model')
    parser.add_argument('--backend',
                        default='fbgemm',
                        help='fbgemm or qnnpack')
    parser.add_argument('--device',
                        default='cuda',
                        help='device')

    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        help='batch size for calibration/training')
    parser.add_argument('--eval-batch-size', default=128, type=int,
                        help='batch size for evaluation')
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num-observer-update-epochs',
                        default=4, type=int, metavar='N',
                        help='number of total epochs to update observers')
    parser.add_argument('--num-batch-norm-update-epochs', default=3,
                        type=int, metavar='N',
                        help='number of total epochs to update batch norm stats')
    parser.add_argument('--num-calibration-batches',
                        default=32, type=int, metavar='N',
                        help='number of batches of training set for \
                              observer calibration ')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr',
                        default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency')
    parser.add_argument(
        '--output-dir', default='./weights_lightnn/', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. \
             It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--post-training-quantize",
        dest="post_training_quantize",
        help="Post training quantize the model",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url',
                        default='env://',
                        help='url used to set up distributed training')

    # debug
    parser.add_argument('--debug', action='store_true',
                        help='Debug Mode, can change dataset by val_portion and train_position')
    parser.add_argument('--portion', type=float,default=0.1, help='portion of training data')
    parser.add_argument('--eval-freq', default=5,type=int,help='evaluate frequency')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
