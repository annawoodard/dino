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
import os
import torch.multiprocessing as mp
import argparse
import json
import logging
from pathlib import Path

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms

from maicara.data.constants import CHIMEC_MEAN, CHIMEC_STD
from chimec import get_datasets

import utils
import vision_transformer as vits
import torch.distributed as dist
from pycox.models.loss import CoxPHLoss
from metrics import predict_coxph_surv, concordance_index, auc


def eval_finetune(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    logger = utils.setup_logging(args.rank, args.output_dir, "finetune")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    in_chans = utils.get_input_channels(args.pretrained_weights)

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](
            patch_size=args.patch_size, num_classes=0, in_chans=in_chans
        )
        embed_dim = model.embed_dim * (
            args.n_last_blocks + int(args.avgpool_patchtokens)
        )
    else:
        logger.error(f"Unknow architecture: {args.arch}")
    model.cuda()
    model.eval()
    # load weights to evaluate
    fit_metadata, test_metadata = utils.load_pretrained_checkpoint(
        model, args.pretrained_weights, args.checkpoint_key
    )
    logger.info(f"Model {args.arch} built.")

    mlp = BilateralMLP(
        embed_dim * 2,
        model,
        args.n_last_blocks,
        hidden_features=512,
        out_features=1,
        act_layer=nn.GELU,
        drop=0.0,
        avgpool=args.avgpool_patchtokens,
    )
    mlp = mlp.cuda()
    mlp = nn.parallel.DistributedDataParallel(
        mlp, device_ids=[args.gpu], find_unused_parameters=True
    )

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(CHIMEC_MEAN, CHIMEC_STD),
        ]
    )
    train_transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(448, interpolation=3),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(CHIMEC_MEAN, CHIMEC_STD),
        ]
    )
    if args.evaluate:
        ## TODO add train outputs/durations/events to checkpoint; these are needed for evaluation
        raise NotImplementedError
        utils.load_pretrained_linear_weights(mlp, args.arch, args.patch_size)
        test_stats = validate_network(test_loader, mlp, c_index)
        logger.info(
            f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    # set optimizer
    optimizer = torch.optim.SGD(
        mlp.parameters(),
        args.lr
        * (args.batch_size_per_gpu * utils.get_world_size())
        / 256.0,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0
    )

    # Optionally resume from a checkpoint
    to_restore = {
        "epoch": 0,
        "best_c_index": 0.0,
        "fold": 0,
        "fit_loader": None,
        "test_loader": None,
    }
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        restore_objects=to_restore,
        state_dict=mlp,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_c_index = to_restore["best_c_index"]
    fit_loaders = to_restore["fit_loader"]
    test_loader = to_restore["test_loader"]
    fold = to_restore["fold"]

    if fit_loaders is None:
        fit_datasets, test_dataset = get_datasets(
            args.prescale,
            train_transform,
            val_transform,
            test_metadata,
            fit_metadata,
            args.folds,
            args.seed,
        )
        fit_loaders = [
            (
                torch.utils.data.DataLoader(
                    train_dataset,
                    sampler=torch.utils.data.distributed.DistributedSampler(
                        train_dataset
                    ),
                    batch_size=args.batch_size_per_gpu,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                ),
                torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size_per_gpu,
                    num_workers=0,
                    pin_memory=True,
                ),
            )
            for train_dataset, val_dataset in fit_datasets
        ]

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=0,
            pin_memory=True,
        )
    logger.info(f"Data loaded with {len(test_dataset)} test exams")
    for i, (train_dataset, val_dataset) in enumerate(fit_datasets):
        logger.info(
            f"Fold {i}: {len(train_dataset)} training exams and {len(val_dataset)} val exams"
        )

    for i, (train_loader, val_loader) in enumerate(fit_loaders):
        if i < fold:
            continue
        early_stopping = utils.EarlyStopping(patience=args.patience)
        for epoch in range(start_epoch, args.epochs):
            train_loader.sampler.set_epoch(epoch)

            train_stats, train_outputs, train_years_to_event, train_events = train(
                mlp, optimizer, train_loader, epoch
            )
            scheduler.step()

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            if utils.is_main_process():
                with (Path(args.output_dir) / f"fold_{i}_train_log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": mlp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_c_index": best_c_index,
                    "fit_loaders": fit_loaders,
                    "test_loader": test_loader,
                    "fold": i,
                }
                torch.save(
                    save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar")
                )
            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                if epoch >= args.val_start:
                    if utils.is_main_process():
                        val_stats = validate_network(
                            val_loader,
                            mlp,
                            train_outputs,
                            train_years_to_event,
                            train_events,
                            args.output_dir,
                        )
                        early_stopping(
                            val_stats["loss"]
                        )  # TODO stop on c index? It is noisier than loss
                        if early_stopping.early_stop:
                            break
                        logger.info(
                            f"Concordance index at epoch {epoch} of the network on the {len(val_loader) * args.batch_size_per_gpu} test images: {val_stats['c_index']:.3f}"
                        )
                        if best_c_index < val_stats["c_index"]:
                            best_c_index = val_stats["c_index"]
                            save_dict = {
                                "epoch": epoch + 1,
                                "state_dict": mlp.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "best_c_index": best_c_index,
                                "fit_loaders": fit_loaders,
                                "test_loader": test_loader,
                                "fold": i,
                            }
                            torch.save(
                                save_dict,
                                os.path.join(
                                    args.output_dir, "best_checkpoint.pth.tar"
                                ),
                            )
                        logger.info(f"Max c-index so far: {best_c_index:.3f}")
                        log_stats = {
                            **{k: v for k, v in log_stats.items()},
                            **{f"val_{k}": v for k, v in val_stats.items()},
                        }
                        with (Path(args.output_dir) / f"fold_{i}_val_log.txt").open(
                            "a"
                        ) as f:
                            f.write(json.dumps(log_stats) + "\n")
        # FIXME do not test every fold, need to stop and retrain on tuned HPs with no val
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "best_checkpoint.pth.tar"),
            state_dict=mlp,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if utils.is_main_process():
            test_stats = validate_network(
                test_loader,
                mlp,
                train_outputs,
                train_years_to_event,
                train_events,
                args.output_dir,
            )
            logger.info(
                f"Concordance index of the best-performing network in the validation set on the {len(test_dataset)} test exams: {test_stats['c_index']:.3f}"
            )
            log_stats = {
                **{k: v for k, v in log_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }
            with (Path(args.output_dir) / "test_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    logger.info(
        f"Training of the supervised MLP on frozen features completed.\n"
        f"Best validation concordance index: {best_c_index:.3f}"
    )


def train(mlp, optimizer, loader, epoch):
    logger = logging.getLogger()
    mlp.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    outputs = []
    durations = []
    events = []
    for (
        image,
        contralateral_image,
        years_to_event,
        event,
        _,
    ) in metric_logger.log_every(loader, 20, header):
        image = image.cuda(non_blocking=True)
        contralateral_image = contralateral_image.cuda(non_blocking=True)
        years_to_event = years_to_event.cuda(non_blocking=True)
        event = event.cuda(non_blocking=True)

        # forward
        output = mlp(image, contralateral_image)

        loss = CoxPHLoss()(output, years_to_event, event)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        outputs += [output]
        durations += [years_to_event]
        events += [event]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    # outputs = torch.tensor(outputs, dtype=torch.float64, device="cuda")
    # needed for estimating censorship distribution
    outputs = torch.cat(outputs[:500])
    durations = torch.cat(durations[:500])
    events = torch.cat(events[:500])
    # dist.barrier()
    # dist.all_reduce(outputs)
    # dist.all_reduce(years_to_event)
    # dist.all_reduce(events)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        outputs,
        durations,
        events,
    )


@torch.no_grad()
def validate_network(
    val_loader,
    mlp,
    train_outputs,
    train_years_to_event,
    train_events,
    output_dir,
):
    logger = logging.getLogger()
    mlp.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    outputs = []
    durations = []
    events = []
    for (
        image,
        contralateral_image,
        years_to_event,
        event,
        study_id,
    ) in metric_logger.log_every(val_loader, 1, header):
        image = image.cuda(non_blocking=False)
        contralateral_image = contralateral_image.cuda(non_blocking=False)
        years_to_event = years_to_event.cuda(non_blocking=False)
        event = event.cuda(non_blocking=False)

        output = mlp(image, contralateral_image)

        loss = CoxPHLoss()(output, years_to_event, event)

        metric_logger.update(loss=loss.item())

        outputs += [output]
        durations += [years_to_event]
        events += [event]

    outputs = torch.cat(outputs)
    durations = torch.cat(durations)
    events = torch.cat(events)

    surv = predict_coxph_surv(
        train_outputs,
        train_years_to_event,
        train_events,
        outputs,
        output_dir=output_dir,
        sample=1.0,
    )
    c_index = concordance_index(surv, durations, events)
    aucs = auc(surv, durations, events)
    metric_logger.update(c_index=c_index)
    for key, value in aucs.items():
        metric_logger.update(key=value)

    logger.info(
        "* c-index {c_index:.3f} loss {losses.global_avg:.3f}".format(
            c_index=c_index,
            losses=metric_logger.loss,
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class BilateralMLP(nn.Module):
    def __init__(
        self,
        in_features,
        encoder,
        n_last_blocks,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        avgpool=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_last_blocks = n_last_blocks
        self.avgpool = avgpool
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1.weight.data.normal_(mean=0.0, std=0.01)
        self.fc1.bias.data.zero_()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2.weight.data.normal_(mean=0.0, std=0.01)
        self.fc2.bias.data.zero_()
        self.drop = nn.Dropout(drop)

    @torch.no_grad()
    def encode(self, image, contralateral_image):
        with torch.no_grad():
            image_intermediate_output = self.encoder.get_intermediate_layers(
                image, self.n_last_blocks
            )
            image_output = torch.cat(
                [x[:, 0] for x in image_intermediate_output], dim=-1
            )
            contra_intermediate_output = self.encoder.get_intermediate_layers(
                contralateral_image, self.n_last_blocks
            )
            contra_output = torch.cat(
                [x[:, 0] for x in contra_intermediate_output], dim=-1
            )
            if self.avgpool:
                image_output = torch.cat(
                    (
                        image_output.unsqueeze(-1),
                        torch.mean(
                            image_intermediate_output[-1][:, 1:], dim=1
                        ).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                contra_output = torch.cat(
                    (
                        contra_output.unsqueeze(-1),
                        torch.mean(
                            contra_intermediate_output[-1][:, 1:], dim=1
                        ).unsqueeze(-1),
                    ),
                    dim=-1,
                )
                image_output = image_output.reshape(image_output.shape[0], -1)
                contra_output = contra_output.reshape(contra_output.shape[0], -1)
            x = torch.cat([image_output, contra_output], dim=1)
            x = x.view(x.size(0), -1)

        return x

    def forward(self, image, contralateral_image):
        x = self.encode(image, contralateral_image)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune")
    parser.add_argument("--seed", default=None, type=int, help="Random seed.")
    parser.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""",
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
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
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
        "--prescale",
        default=1.0,
        type=float,
        help="""Only use PRESCALE percent of data (for development).""",
    )
    parser.add_argument(
        "--sample",
        default=None,
        type=float,
        help="Calculate baseline hazards with SAMPLE subset",
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
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
        "--val_start", default=0, type=int, help="Start validating at epoch VAL_START"
    )
    parser.add_argument(
        "--folds", default=6, type=int, help="Number of validation folds"
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path to save logs and checkpoints"
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
    args = parser.parse_args()
    if args.seed is None:
        args.seed = torch.randint(0, 100000, (1,)).item()
    if args.devices is None:
        args.devices = [f"{i}" for i in range(torch.cuda.device_count())]
    if args.world_size is None:
        args.world_size = len(args.devices)
    args.port = str(utils.get_unused_local_port())
    for fold in range(args.folds):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    mp.spawn(
        eval_finetune,
        nprocs=len(args.devices),
        args=(args,),
    )
