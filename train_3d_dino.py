# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import pickle
import torch.multiprocessing as mp
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import logging

import numpy as np
from PIL import Image
import monai
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from cmmd import CMMDRandomTileDataset

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
import efficientformer
import monai.transforms
from ispy2 import ISPY2Dataset, load_metadata

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("DINO", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "xcit",
            "deit_tiny",
            "deit_small",
            "efficientformer_l3",
            "efficientformer_l3_narrow",
            "efficientformer_l1",
            "efficientformer_l7",
        ]
        + torchvision_archs,
        # + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--out_dim",
        default=65536,
        type=int,
        help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=0,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Per-GPU batch-size: number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", default=800, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--patience",
        default=8,
        type=int,
        help="How many epochs to wait before stopping when loss is not improving",
    )
    parser.add_argument(
        "--freeze_last_layer",
        default=1,
        type=int,
        help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument(
        "--note",
        default="",
        type=str,
        help="""Note to record""",
    )
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="stochastic depth rate"
    )

    parser.add_argument(
        "--include_series",
        type=str,
        nargs="+",
        # default=("unilateral_dce", "bilateral_dce"),
        default=["unilateral_dce"],
        help="""Series to include for training.""",
    )
    # Multi-crop parameters
    parser.add_argument(
        "--tiles_per_image",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--global_crops_size",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(1.0, 1.5),
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        # default=(0.05, 0.4),
        default=(1.5, 2.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_size",
        default=64,
        type=int,
        help="""Size in pixels to divide input image for tile-based training""",
    )
    # Misc
    parser.add_argument(
        "--output_dir", default=".", type=str, help="Path to save logs and checkpoints."
    )
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=5,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--test_size",
        default=0.15,
        type=float,
        help="Proportion of finetuning dataset to hold out for testing; these patients will be excluded from pretraining dataset to avoid data leakage",
    )
    parser.add_argument(
        "--prescale",
        default=1.0,
        type=float,
        help="""Only use PRESCALE percent of data (for development).""",
    )
    parser.add_argument(
        "--stack_views",
        type=utils.bool_flag,
        default=False,
        help="""Whether or not to stack CC and MLO views in the color channel""",
    )
    parser.add_argument(
        "--debug",
        type=utils.bool_flag,
        default=False,
        help="""Run with small debug dataset""",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )
    parser.add_argument(
        "--world_size", default=None, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="which devices to use on local machine",
    )
    parser.add_argument(
        "--saveckp_freq", default=10, type=int, help="Save checkpoint every x epochs."
    )
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    logger = utils.setup_logging(args.output_dir, "pretrain", args.rank)
    logger.info(f"starting training on {os.environ['HOSTNAME']}")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    logger.info(f"called with: python {' '.join(sys.argv)}")
    if utils.is_main_process():
        with (Path(args.output_dir) / "args.pkl").open("wb") as f:
            pickle.dump(args, f)
        utils.log_code_state(args.output_dir)

    # ============ building student and teacher networks ... ============
    in_chans = 2 if args.stack_views else 1
    if args.arch == "densenet":
        # TODO unhardcode
        embed_dim = 256
        student = monai.networks.nets.DenseNet121(
            spatial_dim=3, in_channels=in_chans, out_chan=embed_dim
        )
        teacher = monai.networks.nets.DenseNet121(
            spatial_dim=3, in_channels=in_chans, out_chan=embed_dim
        )
    else:
        logger.error(f"Unknow architecture: {args.arch}")
    logger.info(
        "number of parameters: {:d}".format(
            sum(p.numel() for p in student.parameters() if p.requires_grad)
        )
    )

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info(f"Student and teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number
        + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        # to use with convnet and large batches
        optimizer = utils.LARS(params_groups)
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ optionally resume training ... ============
    to_restore = {
        "epoch": 0,
        "fit_metadata": None,
        "test_metadata": None,
    }
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        restore_objects=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    fit_metadata = to_restore["fit_metadata"]
    test_metadata = to_restore["test_metadata"]

    # ============ preparing data ... ============
    # exclude any patients in testing set from pretraining set
    # fit = val + train sets
    if fit_metadata is None:
        fit_metadata, test_metadata = load_metadata(args.test_size)

    image_level_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=args.include_series),
            monai.transforms.EnsureChannelFirstd(keys=args.include_series),
            monai.transforms.ScaleIntensityd(
                keys=args.include_series,
            ),
        ]
    )
    dataset = ISPY2Dataset(
        transform=image_level_transform,
        exclude=test_metadata.study_id,
        prescale=args.prescale,
        debug=args.debug,
        metadata=fit_metadata,
        include_series=args.include_series,
        # require_series=None,
        # timepoints=None,
        # cache_num=24,
        # cache_rate=1.0,
        # num_workers=4,
    )
    # tile-level transforms
    transform = DataAugmentationDINO(
        args.include_series,
        args.global_crops_scale,
        args.global_crops_size,
        args.local_crops_scale,
        args.local_crops_number,
        args.local_crops_size,
    )
    crop_sampler = monai.transforms.RandSpatialCropSamples(
        roi_size=(
            args.global_crops_size,
            args.global_crops_size,
            args.global_crops_size,
        ),
        num_samples=args.tiles_per_image,
        random_center=True,
        random_size=False,
    )
    patch_dataset = monai.data.PatchDataset(
        dataset=dataset,
        patch_func=crop_sampler,
        samples_per_image=args.tiles_per_image,
        transform=transform,
    )
    shuffle_dataset = monai.data.ShuffleBuffer(patch_dataset, buffer_size=200, seed=0)
    sampler = torch.utils.data.DistributedSampler(shuffle_dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        shuffle_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    utils.write_example_dino_augs(dataset, args.output_dir)

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    logger.info(f"Loss, optimizer and schedulers ready.")

    early_stopping = utils.EarlyStopping(patience=args.patience)

    log_writer = SummaryWriter(os.path.join(args.output_dir, "logs"))

    # TODO add to checkpoint
    start_time = time.time()
    logger.info("Starting DINO training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats, last_teacher_output, last_student_output = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
            logger,
        )
        if (args.tile_size != None) and (args.dataset == "chimec"):
            dataset.load_views()
        early_stopping(train_stats["loss"])
        if early_stopping.early_stop:
            break

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
            "fit_metadata": fit_metadata,
            "test_metadata": test_metadata,
            "in_chans": in_chans,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            for key, value in train_stats.items():
                log_writer.add_scalar(f"train/{key}", value, epoch)
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            log_writer.add_histogram(
                "teacher_representation", last_teacher_output, epoch
            )
            log_writer.add_histogram(
                "student_representation", last_student_output, epoch
            )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
    logger,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(
                images[:2]
            )  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(
                student.module.parameters(), teacher_without_ddp.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        teacher_output,
        student_output,
    )


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class DataAugmentationDINO(object):
    def __init__(
        self,
        include_series,
        global_crops_scale,
        global_crops_size,
        local_crops_scale,
        local_crops_number,
        local_crops_size,
    ):
        patch_level_transform = monai.transforms.Compose(
            [
                monai.transforms.SavitzkyGolaySmoothd(
                    keys=include_series, window_length=9, order=1
                ),
                monai.transforms.OneOf(
                    transforms=[
                        monai.transforms.RandGaussianNoised(
                            keys=include_series, prob=1.0
                        ),
                        monai.transforms.RandBiasFieldd(
                            keys=include_series, prob=1.0, coeff_range=(0.2, 0.3)
                        ),
                        monai.transforms.RandKSpaceSpikeNoised(
                            keys=include_series, intensity_range=(11, 13), prob=1.0
                        ),
                        monai.transforms.Identityd(keys=include_series),
                    ]
                ),
                monai.transforms.OneOf(
                    transforms=[
                        monai.transforms.SavitzkyGolaySmoothd(
                            keys=include_series, window_length=3, order=1
                        ),
                        monai.transforms.RandHistogramShiftd(
                            keys=include_series, prob=1.0, num_control_points=(9, 11)
                        ),
                        monai.transforms.RandGaussianSmoothd(
                            keys=include_series, prob=1.0
                        ),
                        monai.transforms.Identityd(keys=include_series),
                    ],
                ),
                monai.transforms.OneOf(
                    transforms=[
                        monai.transforms.Rand3DElasticd(
                            keys=include_series,
                            sigma_range=(5, 7),
                            magnitude_range=(50, 350),
                            prob=1.0,
                        ),
                        monai.transforms.RandAffineD(
                            keys=include_series,
                            shear_range=(0.0, 0.2),
                            rotate_range=(0, 45, 0),
                            prob=1.0,
                        ),
                        monai.transforms.Identityd(keys=include_series),
                    ]
                ),
                monai.transforms.OneOf(
                    transforms=[
                        monai.transforms.RandFlipd(keys=include_series),
                        monai.transforms.RandAxisFlipd(keys=include_series),
                    ],
                ),
                monai.transforms.RandRotated(
                    keys=include_series, range_x=90, range_y=90, range_z=90
                ),
            ]
        )
        self.global_transform_1 = monai.transforms.Compose(
            [
                monai.transforms.RandZoomd(
                    keys=include_series,
                    min_zoom=global_crops_scale[0],
                    max_zoom=global_crops_scale[1],
                ),
                monai.transforms.RandSpatialCropd(
                    keys=include_series,
                    roi_size=(global_crops_size, global_crops_size, global_crops_size),
                    random_size=False,
                ),
                patch_level_transform,
            ]
        )
        self.global_transform_2 = monai.transforms.Compose(
            [
                monai.transforms.RandZoomd(
                    keys=include_series,
                    min_zoom=global_crops_scale[0],
                    max_zoom=global_crops_scale[1],
                ),
                monai.transforms.RandSpatialCropd(
                    keys=include_series,
                    roi_size=(global_crops_size, global_crops_size, global_crops_size),
                    random_size=False,
                ),
                monai.transforms.RandShiftIntensityd(
                    keys=include_series, offsets=(10, 20)
                ),
                patch_level_transform,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = monai.transforms.Compose(
            [
                monai.transforms.RandZoomd(
                    keys=include_series,
                    min_zoom=local_crops_scale[0],
                    max_zoom=local_crops_scale[1],
                ),
                monai.transforms.RandSpatialCropd(
                    keys=include_series,
                    roi_size=(local_crops_size, local_crops_size, local_crops_size),
                    random_size=False,
                ),
                patch_level_transform,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform_1(image))
        crops.append(self.global_transform_2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINO", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.seed is None:
        args.seed = torch.randint(0, 100000, (1,)).item()
    # args.port = str(utils.get_unused_local_port())
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
