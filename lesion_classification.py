import argparse
import torch.nn.functional as F
import torchmetrics
import sys
from torchvision import models as torchvision_models
import matplotlib.pyplot as plt
import time
import pandas as pd
import json
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from monai.networks.nets import milmodel
from transforms import GridTile, RandGridTile

torch.autograd.set_detect_anomaly(True)
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from maicara.data.constants import CHIMEC_MEAN, CHIMEC_STD
from pycox.models.loss import CoxPHLoss
from torch import nn
import efficientformer
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as pth_transforms
from torchvision.utils import make_grid

import utils
import vision_transformer as vits
from metrics import auc
from cmmd import get_datasets


def train_mil(gpu, result_queue, fold_queue, args):
    torch.cuda.set_device(gpu)
    if args.seed is None:
        args.seed = torch.randint(0, 100000, (1,)).item()
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    while fold_queue.qsize() > 0:
        fold = fold_queue.get()
        logger = utils.setup_logging(args.output_dir, f"fold_{fold}")
        logger.info(f"Processing fold {fold} on gpu {gpu}")
        logger.info(
            "\n".join(
                "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
            )
        )
        log_writer = SummaryWriter(
            os.path.join(
                args.output_dir, "logs", "test" if fold == None else f"fold_{fold}"
            )
        )

        datasets = torch.load(os.path.join(args.output_dir, "datasets.pth.tar"))
        fit_datasets = datasets["fit_datasets"]
        test_dataset = datasets["test_dataset"]
        train_dataset, val_dataset = fit_datasets[fold if fold is not None else 0]
        if args.fix_fold != None:
            train_dataset, val_dataset = fit_datasets[args.fix_fold]

        if fold == None:
            # before evaluating the testing dataset we train with all fit data
            fit_dataset = train_dataset + val_dataset
        else:
            fit_dataset = train_dataset
            test_dataset = val_dataset

        fit_loader = torch.utils.data.DataLoader(
            fit_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        # ============ building network ... ============
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.arch in vits.__dict__.keys():
            encoder = vits.__dict__[args.arch](
                patch_size=args.patch_size, num_classes=0
            )
            embed_dim = encoder.embed_dim * (
                args.n_last_blocks + int(args.avgpool_patchtokens)
            )
        # if the network is a XCiT
        elif "xcit" in args.arch:
            encoder = torch.hub.load(
                "facebookresearch/xcit:main", args.arch, num_classes=0
            )
            embed_dim = encoder.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif args.arch in torchvision_models.__dict__.keys():
            encoder = torchvision_models.__dict__[args.arch]()
            if "resnet" in args.arch:
                # make grayscale
                encoder.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            embed_dim = encoder.fc.weight.shape[1]
        elif args.arch in efficientformer.__dict__.keys():
            encoder = efficientformer.__dict__[args.arch](distillation=False)
            embed_dim = encoder.embed_dims[-1]
        else:
            print(f"Unknow architecture: {args.arch}")
            sys.exit(1)
        encoder.cuda()
        encoder.eval()
        # load pretrained weights for feature extraction
        utils.load_pretrained_checkpoint(
            encoder, args.pretrained_weights, args.checkpoint_key
        )
        if args.freeze_encoder:
            encoder.requires_grad_(False)
        logger.info(f"Model {args.arch} built.")
        model = milmodel.MILModel(
            num_classes=2,
            backbone=encoder,
            backbone_num_features=embed_dim,
            mil_mode=args.mil_mode,
        )
        print(model)
        model = model.cuda()

        if args.evaluate:
            ## TODO
            raise NotImplementedError
            utils.load_pretrained_linear_weights(model, args.arch, args.patch_size)
            test_stats = validate_network(test_loader, model, c_index)
            logger.info(
                f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%"
            )
            return

        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr
            * (args.batch_size * utils.get_world_size())
            / 256.0,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0
        )
        best_auc = 0.0

        early_stopping = utils.EarlyStopping(patience=args.patience)
        for epoch in range(0, args.epochs):
            train_stats = train(
                model,
                optimizer,
                fit_loader,
                epoch,
                args.batch_size,
                args.avgpool_patchtokens,
                args.arch,
            )
            scheduler.step()

            for key, value in train_stats.items():
                log_writer.add_scalar(f"train/{key}", value, epoch)
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            with (Path(args.output_dir) / f"fold_{fold}_train_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if fold != None:
                if (epoch % args.val_freq == 0) or (epoch == args.epochs - 1):
                    val_stats = validate_network(
                        test_loader,
                        model,
                        args.n_last_blocks,
                        args.avgpool_patchtokens,
                        args.arch,
                    )

                    for key, value in val_stats.items():
                        log_writer.add_scalar(f"val/{key}", value, epoch)

                    early_stopping(val_stats["loss"])
                    if args.early_stop and early_stopping.early_stop:
                        break
                    logger.info(
                        f"Accuracy at epoch {epoch} of the network on the {len(val_dataset)} test images: {val_stats['acc']:.3f}"
                    )
                    # if best_c_index < val_stats["c_index"]:
                    #     best_c_index = val_stats["c_index"]
                    # logger.info(f"Max c-index so far: {best_c_index:.3f}")
                    log_stats = {
                        **{k: v for k, v in log_stats.items()},
                        **{f"val_{k}": v for k, v in val_stats.items()},
                    }
                    with (Path(args.output_dir) / f"fold_{fold}_val_log.txt").open(
                        "a"
                    ) as f:
                        f.write(json.dumps(log_stats) + "\n")
        if fold == None:
            test_stats = validate_network(
                test_loader,
                model,
                args.n_last_blocks,
                args.avgpool_patchtokens,
                args.arch,
            )
            logger.info(
                f"Accuracy of the model trained with (train + val) datasets on the {len(test_loader)} test views: {test_stats['acc']:.3f}"
            )

            for key, value in test_stats.items():
                log_writer.add_scalar(f"test/{key}", value, epoch)
            log_stats = {
                **{k: v for k, v in log_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }
            with (Path(args.output_dir) / "test_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # to conserve space, do not save a checkpoint if this is a sweep
            if args.project is None:
                save_dict = {
                    "state_dict": model.state_dict(),
                }
                torch.save(
                    save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar")
                )

        logger.info(
            f"Training of the supervised model on frozen features completed.\n"
            # f"Best validation concordance index: {best_c_index:.3f}"
        )
        result = test_stats if fold == None else val_stats
        result["fold"] = fold

        result_queue.put(result)


def train(model, optimizer, loader, epoch, n, avgpool, arch):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    acc = torchmetrics.Accuracy(num_classes=2).cuda()
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        # with torch.no_grad():
        if "vit" in arch:
            intermediate_output = model.net.get_intermediate_layers(inp, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool:
                output = torch.cat(
                    (
                        output.unsqueeze(-1),
                        torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                output = output.reshape(output.shape[0], -1)
        else:
            output = model(inp)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)
        acc.update(output, target)
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN!")

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        # torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    acc = acc.compute()
    metric_logger.update(acc=acc.item())
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, n, avgpool, arch):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    auc = torchmetrics.AUROC(num_classes=2).cuda()
    acc = torchmetrics.Accuracy(num_classes=2).cuda()
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        one_hot_target = F.one_hot(target, num_classes=2)

        with torch.no_grad():
            if "vit" in arch:
                intermediate_output = model.net.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat(
                        (
                            output.unsqueeze(-1),
                            torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(
                                -1
                            ),
                        ),
                        dim=-1,
                    )
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        loss = nn.CrossEntropyLoss()(output, target)

        auc.update(output, one_hot_target)
        acc.update(output, target)

        metric_logger.update(loss=loss.item())
    auc = auc.compute()
    acc = acc.compute()
    metric_logger.update(auc=auc.item())
    metric_logger.update(acc=acc.item())
    print(
        "* accuracy {acc.global_avg:.3f} loss {losses.global_avg:.3f} auc {auc.global_avg:.3f}".format(
            acc=metric_logger.acc, losses=metric_logger.loss, auc=metric_logger.auc
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser("Train model")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    parser.add_argument(
        "--prescale",
        default=1.0,
        type=float,
        help="""Only use PRESCALE percent of data (for development).""",
    )
    parser.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
    )
    parser.add_argument(
        "--freeze_encoder",
        default=True,
        type=utils.bool_flag,
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""",
    )
    parser.add_argument(
        "--log_images",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to log example images to Tensorboard""",
    )
    parser.add_argument(
        "--early_stop",
        default=True,
        type=utils.bool_flag,
        help="""Enable early stopping.""",
    )
    parser.add_argument("--arch", default="vit_small", type=str, help="Architecture")
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument(
        "--overlap",
        default=0.25,
        type=float,
        help="Patch overlap",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="How many epochs to wait before stopping when val loss is not improving",
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Per-GPU batch-size for mil training",
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
        "--num_workers",
        default=5,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--val_freq", default=1, type=int, help="Epoch frequency for validation."
    )
    parser.add_argument(
        "--fix_fold", default=None, type=int, help="Fix all folds to fold FIX_FOLD."
    )
    parser.add_argument(
        "--val_start", default=0, type=int, help="Start validating at epoch VAL_START"
    )
    parser.add_argument(
        "--folds", default=5, type=int, help="Number of validation folds"
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path to save logs and checkpoints"
    )
    parser.add_argument(
        "--config", default=None, type=str
    )  # TODO implement config loading from yaml
    parser.add_argument(
        "--autolabel",
        default=False,
        help="Make final output directory in an automatically labeled directory under OUTPUT_DIR",
        type=utils.bool_flag,
    )
    parser.add_argument(
        "--tile_size",
        default=(224, 224),
        nargs="+",
        type=int,
        help="""Size in pixels to divide input image for tile-based training""",
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
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
    parser.add_argument("--mil_mode", default="att_trans", help="MIL algorithm")
    parser.add_argument("--project", default=None, type=str, help="wandb project")
    args = parser.parse_args()
    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))
    if args.world_size is None:
        args.world_size = len(args.devices)
    args.port = str(utils.get_unused_local_port())
    args.output_dir = utils.prepare_output_dir(args.output_dir, args.autolabel)
    utils.log_code_state(args.output_dir)

    logger = utils.setup_logging(args.output_dir, f"fold_")
    if not os.path.isfile(os.path.join(args.output_dir, "datasets.pth.tar")):
        val_transform = pth_transforms.Compose(
            [
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(CHIMEC_MEAN, CHIMEC_STD),
                GridTile(tile_size=args.tile_size, max_frac_black=0.96),
            ]
        )
        train_transform = pth_transforms.Compose(
            [
                pth_transforms.RandomHorizontalFlip(),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(CHIMEC_MEAN, CHIMEC_STD),
                RandGridTile(tile_size=args.tile_size, max_frac_black=0.96),
            ]
        )

        fit_datasets, test_dataset = get_datasets(
            args.prescale,
            train_transform,
            val_transform,
            args.folds,
            args.seed,
        )
        torch.save(
            {"fit_datasets": fit_datasets, "test_dataset": test_dataset},
            os.path.join(args.output_dir, "datasets.pth.tar"),
        )
        logger.info(f"Data loaded")
        for i, (train_dataset, val_dataset) in enumerate(fit_datasets):
            logger.info(
                f"Fold {i}: {len(train_dataset)} training views and {len(val_dataset)} val views"
            )

    # We may not have a matching number of folds and GPUs
    # so use a queue and a while loop to process all folds
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    fold_queue = manager.Queue()
    for fold in range(args.folds):
        fold_queue.put(fold)
    # On None, we combine train and val datasets to train and test on held-out test dataset
    fold_queue.put(None)

    mp.spawn(
        train_mil,
        nprocs=len(args.devices),
        args=(result_queue, fold_queue, args),
        join=True,
    )
    results = [result_queue.get() for i in range(result_queue.qsize())]
    results = pd.DataFrame(results)
    logger.info(
        "training completed in {:.1f} minutes with mean validation accuracy {:.3f} +/ {:.3f}; test accuracy {:.3f}".format(
            (time.time() - start) / 60.0,
            results.acc.mean(),
            results.acc.std(),
            results[pd.isnull(results.fold)].c_index.iloc[0],
        )
    )

    results.to_pickle(Path(args.output_dir) / "results.pkl")
    with (Path(args.output_dir) / "args.json").open("w") as f:
        f.write(json.dumps(vars(args)) + "\n")

    # TODO
    # if args.project is not None:
    #     wandb.login()
    #     config = args.config if args.config else args
    #     run = wandb.init(config=config, project=args.project)
    #     wandb.log(
    #         {
    #             "val_c_index": results[~pd.isnull(results.fold)].c_index.mean(),
    #             "val_c_index_std": results[~pd.isnull(results.fold)].c_index.std(),
    #             "test_c_index": results[pd.isnull(results.fold)].c_index,
    #         }
    #     )
