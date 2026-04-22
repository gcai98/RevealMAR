import argparse
import copy
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from util.crop import center_crop_arr
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.loader import CachedFolder

from models.vae import AutoencoderKL
from models import revealmar
from engine_mar import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('RevealMAR training with Diffusion Loss', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')
    parser.add_argument('--epochs', default=400, type=int)

    parser.add_argument('--model', default='revealmar_large', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default='pretrained_models/vae/kl16.ckpt', type=str,
                        help='VAE checkpoint path')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='VAE output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch.')

    parser.add_argument('--num_iter', default=64, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=1.0, type=float, help='classifier-free guidance')
    parser.add_argument('--cfg_schedule', default='linear', type=str)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=64, help='generation batch size')

    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='learning rate schedule')
    parser.add_argument('--warmup_epochs', type=int, default=100, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--ema_rate', default=0.9999, type=float)

    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)

    parser.add_argument('--diffloss_d', type=int, default=12)
    parser.add_argument('--diffloss_w', type=int, default=1536)
    parser.add_argument('--num_sampling_steps', type=str, default='100')
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float, help='diffusion loss sampling temperature')

    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--class_num', default=1000, type=int)

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5, help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_cached', action='store_true', dest='use_cached',
                        help='Use cached latents')
    parser.add_argument('--cached_path', default='', help='path to cached latents')

    parser.add_argument('--planner_hidden_dim', default=128, type=int,
                        help='hidden dimension for RevealMAR planner')
    parser.add_argument('--planner_loss_weight', default=1.0, type=float,
                        help='loss weight for RevealMAR planner')
    parser.add_argument('--candidate_pool_size', default=8, type=int,
                        help='candidate pool size for RevealMAR')
    parser.add_argument('--pseudo_target_type', default='none', type=str,
                        choices=['none', 'gt_reveal', 'pred_reveal', 'mixed_reveal'],
                        help='pseudo target variant for RevealMAR scaffolding')
    parser.add_argument('--sampling_policy', default='baseline', type=str,
                        choices=['baseline', 'planner'],
                        help='sampling policy for RevealMAR evaluation/sampling')
    parser.add_argument('--candidate_selection_mode', default='topk', type=str,
                        choices=['topk', 'mixed'],
                        help='candidate subset proposal mode for RevealMAR')
    parser.add_argument('--budget_mode', default='soft', type=str,
                        choices=['soft', 'hard'],
                        help='budget mode for RevealMAR')
    parser.add_argument('--mixed_policy_ratio', default=0.0, type=float,
                        help='mixed policy ratio for RevealMAR')

    parser.add_argument('--log_revealmar_losses', action='store_true',
                        help='Print RevealMAR loss decomposition (total, diffloss, planner_aux) during training')
    parser.add_argument('--log_revealmar_loss_freq', type=int, default=20,
                        help='Print every N forward passes when --log_revealmar_losses is set')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_cached:
        dataset_train = CachedFolder(args.cached_path)
    else:
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    vae = AutoencoderKL(embed_dim=args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).cuda().eval()
    for param in vae.parameters():
        param.requires_grad = False

    model = revealmar.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
        planner_hidden_dim=args.planner_hidden_dim,
        planner_loss_weight=args.planner_loss_weight,
        candidate_pool_size=args.candidate_pool_size,
        pseudo_target_type=args.pseudo_target_type,
        sampling_policy=args.sampling_policy,
        candidate_selection_mode=args.candidate_selection_mode,
        budget_mode=args.budget_mode,
        mixed_policy_ratio=args.mixed_policy_ratio,
    )

    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))

    model.to(device)
    model_without_ddp = model

    model_params = list(model_without_ddp.parameters())
    ema_params = copy.deepcopy(model_params)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_without_ddp._revealmar_loss_log_freq = (
        int(args.log_revealmar_loss_freq) if args.log_revealmar_losses else 0
    )

    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if args.resume and os.path.exists(os.path.join(args.resume, 'checkpoint-last.pth')):
        checkpoint = torch.load(os.path.join(args.resume, 'checkpoint-last.pth'), map_location='cpu')
        load_result = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if load_result.missing_keys:
            print('Resume missing model keys (likely new RevealMAR params): {}'.format(load_result.missing_keys))
        if load_result.unexpected_keys:
            print('Resume unexpected model keys (ignored): {}'.format(load_result.unexpected_keys))
        if 'model_ema' in checkpoint:
            ema_state_dict = checkpoint['model_ema']
            ema_params_new = []
            ema_missing_keys = []
            for name, param in model_without_ddp.named_parameters():
                if name in ema_state_dict:
                    ema_params_new.append(ema_state_dict[name].to(device))
                else:
                    ema_missing_keys.append(name)
                    ema_params_new.append(param.detach().clone())
            if ema_missing_keys:
                print('EMA missing keys in checkpoint, fallback to current params: {}'.format(ema_missing_keys))
            ema_params = ema_params_new
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        print('Resume checkpoint %s' % args.resume)
        del checkpoint
    else:
        print('Training from scratch')

    if args.evaluate:
        torch.cuda.empty_cache()
        evaluate(model_without_ddp, vae, ema_params, args, 0,
                 batch_size=args.eval_bsz, log_writer=log_writer, cfg=args.cfg, use_ema=True)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            model, vae,
            model_params, ema_params,
            data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                            ema_params=ema_params, epoch_name='last')

        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            evaluate(model_without_ddp, vae, ema_params, args, epoch,
                     batch_size=args.eval_bsz, log_writer=log_writer, cfg=1.0, use_ema=True)
            if not (args.cfg == 1.0 or args.cfg == 0.0):
                evaluate(model_without_ddp, vae, ema_params, args, epoch,
                         batch_size=args.eval_bsz // 2, log_writer=log_writer, cfg=args.cfg, use_ema=True)
            torch.cuda.empty_cache()

        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    main(args)
