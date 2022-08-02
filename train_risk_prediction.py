import argparse
import matplotlib.pyplot as plt
import lifelines
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

torch.autograd.set_detect_anomaly(True)
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from maicara.data.constants import CHIMEC_MEAN, CHIMEC_STD
from pycox.models.loss import CoxPHLoss
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as pth_transforms
from torchvision.utils import make_grid

import utils
import vision_transformer as vits
from chimec import get_datasets, log_summary
from metrics import auc, predict_coxph_surv
from loss import DeepCENTLoss


@torch.no_grad()
def extract_features_and_labels(
    encoder,
    data_loader,
    multiscale=False,
    avgpool=False,
    n_last_blocks=4,
    return_images=False,
):
    logger = logging.getLogger()
    logger.info(f"Extracting features for {len(data_loader.dataset)} samples")
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = []
    years_to_events = []
    events = []
    study_ids = []
    images = []
    contralateral_images = []

    for (
        image,
        contralateral_image,
        years_to_event,
        event,
        study_id,
    ) in metric_logger.log_every(data_loader, 10):
        image = image.cuda(non_blocking=True)
        contralateral_image = contralateral_image.cuda(non_blocking=True)
        # TODO
        # if multiscale:
        #     feats = utils.multi_scale(samples, encoder)
        # else:
        #     feats = model(encoder).clone()

        image_intermediate_output = encoder.get_intermediate_layers(
            image, n_last_blocks
        )
        image_output = torch.cat([x[:, 0] for x in image_intermediate_output], dim=-1)
        contra_intermediate_output = encoder.get_intermediate_layers(
            contralateral_image, n_last_blocks
        )
        contra_output = torch.cat([x[:, 0] for x in contra_intermediate_output], dim=-1)
        if avgpool:
            image_output = torch.cat(
                (
                    image_output.unsqueeze(-1),
                    torch.mean(image_intermediate_output[-1][:, 1:], dim=1).unsqueeze(
                        -1
                    ),
                ),
                dim=-1,
            )
            contra_output = torch.cat(
                (
                    contra_output.unsqueeze(-1),
                    torch.mean(contra_intermediate_output[-1][:, 1:], dim=1).unsqueeze(
                        -1
                    ),
                ),
                dim=-1,
            )
            image_output = image_output.reshape(image_output.shape[0], -1)
            contra_output = contra_output.reshape(contra_output.shape[0], -1)
        x = torch.cat([image_output, contra_output], dim=1)
        x = x.view(x.size(0), -1)

        # TODO check this
        x = nn.functional.normalize(x, dim=1, p=2)

        features += [x]
        years_to_events += [years_to_event]
        events += [event]
        study_ids += [study_id]
        if return_images:
            images += [image]
            contralateral_images += [contralateral_image]

    features = torch.cat(features)
    years_to_events = torch.cat(years_to_events)
    events = torch.cat(events)
    study_ids = torch.cat(study_ids)

    return features, years_to_events, events, study_ids, images, contralateral_images


def train_mlp(gpu, result_queue, fold_queue, in_chans, args):
    torch.cuda.set_device(gpu)
    if args.seed is None:
        args.seed = torch.randint(0, 100000, (1,)).item()
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True
    while fold_queue.qsize() > 0:
        fold = fold_queue.get()
        logger = utils.setup_logging(args.output_dir, f"train_mlp_{fold}")
        logger.info(f"Processing fold {fold} on gpu {gpu}")
        utils.log_code_state(args.output_dir)
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

        train_serialize_path = os.path.join(
            os.path.join(args.features_dir, f"train_features_fold_{fold}.pth")
        )
        test_serialize_path = os.path.join(
            os.path.join(args.features_dir, f"test_features_fold_{fold}.pth")
        )
        datasets = torch.load(os.path.join(args.features_dir, "datasets.pth.tar"))
        fit_datasets = datasets["fit_datasets"]
        test_dataset = datasets["test_dataset"]
        # before evaluating the testing dataset we train with all fit data (train + val)
        train_dataset, val_dataset = fit_datasets[fold if fold is not None else 0]
        log_summary("train", train_dataset.metadata)
        log_summary("val", val_dataset.metadata)
        if args.fold != None:
            train_dataset, val_dataset = fit_datasets[args.fold]

        if fold == None:
            fit_dataset = train_dataset + val_dataset
            return_images = True
            # return_images = args.log_images

        else:
            fit_dataset = train_dataset
            test_dataset = val_dataset
            return_images = False

        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch](
                patch_size=args.patch_size, num_classes=0, in_chans=in_chans
            )
        else:
            logger.error(f"Unknown architecture: {args.arch}")

        model.cuda()
        model.eval()
        # load pretrained weights for feature extraction
        utils.load_pretrained_checkpoint(
            model, args.pretrained_weights, args.checkpoint_key
        )
        logger.info(f"Model {args.arch} built.")
        try:
            (
                train_features,
                train_durations,
                train_events,
                train_study_ids,
                _,
                _,
            ) = torch.load(train_serialize_path)
        except (RuntimeError, FileNotFoundError):
            (
                train_features,
                train_durations,
                train_events,
                train_study_ids,
                _,
                _,
            ) = extract_and_save(
                model,
                fit_dataset,
                args.num_workers,
                args,
                train_serialize_path,
                return_images=False,
            )
        try:
            (
                test_features,
                test_durations,
                test_events,
                test_study_ids,
                test_images,
                test_contralateral_images,
            ) = torch.load(test_serialize_path)
        except (RuntimeError, FileNotFoundError):
            (
                test_features,
                test_durations,
                test_events,
                test_study_ids,
                test_images,
                test_contralateral_images,
            ) = extract_and_save(
                model,
                test_dataset,
                0,
                args,
                test_serialize_path,
                return_images=return_images,
            )
        del model
        mlp = BilateralMLP(
            train_features.shape[-1],
            hidden_features=512,
            out_features=1,
            act_layer=nn.GELU,
            drop=0.0,
        )
        print(mlp)
        mlp = mlp.cuda()

        if args.evaluate:
            ## TODO
            raise NotImplementedError
            utils.load_pretrained_linear_weights(mlp, args.arch, args.patch_size)
            test_stats = validate_network(test_loader, mlp, c_index)
            logger.info(
                f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%"
            )
            return

        optimizer = torch.optim.SGD(
            mlp.parameters(),
            args.lr
            * (args.batch_size * utils.get_world_size())
            / 256.0,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0
        )
        best_c_index = 0.0

        early_stopping = utils.EarlyStopping(patience=args.patience)
        if args.loss == "DeepCENTLoss":
            criterion = DeepCENTLoss(lambda_p=args.lambda_p, lambda_r=args.lambda_r)
        else:
            criterion = CoxPHLoss()
        for epoch in range(0, args.epochs):
            train_stats, train_outputs = train(
                mlp,
                optimizer,
                train_features,
                train_durations,
                train_events,
                epoch,
                args.batch_size,
                criterion,
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
                    (val_stats, _, _,) = validate_network(
                        test_features,
                        test_durations,
                        test_events,
                        mlp,
                        train_outputs[:500].cuda(),
                        train_durations[:500].cuda(),
                        train_events[:500].cuda(),
                        args.output_dir,
                        args.batch_size,
                        criterion,
                    )

                    for key, value in val_stats.items():
                        log_writer.add_scalar(f"val/{key}", value, epoch)

                    early_stopping(val_stats["loss"])
                    if args.early_stop and early_stopping.early_stop:
                        break
                    logger.info(
                        f"Concordance index at epoch {epoch} of the network on the {len(test_features)} test images: {val_stats['c_index']:.3f}"
                    )
                    if best_c_index < val_stats["c_index"]:
                        best_c_index = val_stats["c_index"]
                    logger.info(f"Max c-index so far: {best_c_index:.3f}")
                    log_stats = {
                        **{k: v for k, v in log_stats.items()},
                        **{f"val_{k}": v for k, v in val_stats.items()},
                    }
                    with (Path(args.output_dir) / f"fold_{fold}_val_log.txt").open(
                        "a"
                    ) as f:
                        f.write(json.dumps(log_stats) + "\n")
        if fold == None:
            (test_stats, lowest_loss_indices, highest_loss_indices,) = validate_network(
                test_features,
                test_durations,
                test_events,
                mlp,
                train_outputs[:500].cuda(),
                train_durations[:500].cuda(),
                train_events[:500].cuda(),
                args.output_dir,
                # args.batch_size,
                35,
                criterion,
            )
            logger.info(
                f"Concordance index of the model trained with (train + val) datasets on the {test_features.shape[0]} test exams: {test_stats['c_index']:.3f}"
            )

            def prepare_images(images, contralateral_images, indices):
                result = []
                for i, c in zip(
                    images[indices[0] : indices[1]],
                    contralateral_images[indices[0] : indices[1]],
                ):
                    result += [torch.cat([i.squeeze(0), c.squeeze(0)], dim=1)]

                return result
                # result = (
                #     images[indices[0] : indices[1]]
                #     + contralateral_images[indices[0] : indices[1]]
                # )

                # return [i.squeeze(0) for i in result]

            img_grid = make_grid(
                prepare_images(
                    test_images, test_contralateral_images, highest_loss_indices
                )
            )
            plt.figure(figsize=(20, 40))
            plt.imshow(
                img_grid[
                    0,
                    :,
                ],
                cmap="Greys",
                # interpolation="nearest"
            )
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            print("INDICES!!!!")
            print(highest_loss_indices, lowest_loss_indices)
            plt.savefig(Path(args.output_dir) / "highest_loss_images.pdf")
            plt.savefig(Path(args.output_dir) / "highest_loss_images.png")
            log_writer.add_image("highest_loss_images", img_grid)
            img_grid = make_grid(
                prepare_images(
                    test_images, test_contralateral_images, lowest_loss_indices
                )
            )
            plt.figure(figsize=(20, 40))
            plt.imshow(
                img_grid[
                    0,
                    :,
                ],
                cmap="Greys",
                # interpolation="nearest"
            )
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            plt.savefig(Path(args.output_dir) / "lowest_loss_images.pdf")
            plt.savefig(Path(args.output_dir) / "lowest_loss_images.png")
            log_writer.add_image("lowest_loss_images", img_grid)
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
                    "state_dict": mlp.state_dict(),
                }
                torch.save(
                    save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar")
                )

        logger.info(
            f"Training of the supervised MLP on frozen features completed.\n"
            f"Best validation concordance index: {best_c_index:.3f}"
        )
        result = test_stats if fold == None else val_stats
        result["fold"] = fold

        result_queue.put(result)


def extract_and_save(model, dataset, num_workers, args, path, return_images=False):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.extraction_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    (
        features,
        durations,
        events,
        study_ids,
        images,
        contralateral_images,
    ) = extract_features_and_labels(
        model,
        loader,
        multiscale=False,
        avgpool=args.avgpool_patchtokens,
        n_last_blocks=args.n_last_blocks,
        return_images=return_images,
    )
    torch.save(
        [
            features.cpu(),
            durations.cpu(),
            events.cpu(),
            study_ids.cpu(),
            [i.cpu() for i in images],
            [i.cpu() for i in contralateral_images],
        ],
        path,
    )

    return features, durations, events, study_ids, images, contralateral_images


def train(mlp, optimizer, features, durations, events, epoch, batch_size, criterion):
    logger = logging.getLogger()
    mlp.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    num_samples = features.shape[0]
    logger.info(f"Begin training on {num_samples} samples")
    outputs = []
    lowers = np.arange(0, num_samples, batch_size)[:-1]
    uppers = lowers + batch_size
    # Merge last partial batch with previous batch; loss is undefined with no positive examples
    uppers[-1] = num_samples
    for lower, upper in metric_logger.log_every(list(zip(lowers, uppers)), 19, header):
        batch_features = features[lower:upper, :].cuda(non_blocking=True)
        batch_durations = durations[lower:upper].cuda(non_blocking=True)
        batch_events = events[lower:upper].cuda(non_blocking=True)

        # forward
        output = mlp(batch_features)

        if isinstance(criterion, CoxPHLoss):
            loss = criterion(output, batch_durations, batch_events)
            metric_logger.update(loss=loss.item())
        else:
            mse_loss, penalty_loss, rank_loss = criterion(
                output, batch_durations, batch_events
            )
            loss = mse_loss + penalty_loss + rank_loss
            metric_logger.update(
                mse_loss=mse_loss.item(),
                penalty_loss=penalty_loss.item(),
                rank_loss=rank_loss.item(),
            )
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN!")
        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        outputs += [output]

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: %s", metric_logger)

    # needed for estimating censorship distribution
    outputs = torch.cat(outputs)
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        outputs,
    )


@torch.no_grad()
def validate_network(
    features,
    durations,
    events,
    mlp,
    train_outputs,
    train_years_to_event,
    train_events,
    output_dir,
    batch_size,
    criterion,
):
    logger = logging.getLogger()
    mlp.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    outputs = []
    num_samples = features.shape[0]
    outputs = []
    lowers = np.arange(0, num_samples, batch_size)[:-1]
    uppers = lowers + batch_size
    # Merge last partial batch with previous batch; loss is undefined with no positive examples
    uppers[-1] = num_samples
    highest_loss = -torch.inf
    lowest_loss = torch.inf
    for lower, upper in metric_logger.log_every(list(zip(lowers, uppers)), 20, header):
        batch_features = features[lower:upper, :].cuda(non_blocking=True)
        batch_durations = durations[lower:upper].cuda(non_blocking=True)
        batch_events = events[lower:upper].cuda(non_blocking=True)

        output = mlp(batch_features)

        if isinstance(criterion, CoxPHLoss):
            loss = criterion(output, batch_durations, batch_events)
            metric_logger.update(loss=loss.item())
        else:
            mse_loss, penalty_loss, rank_loss = criterion(
                output, batch_durations, batch_events
            )
            loss = mse_loss + penalty_loss + rank_loss
            metric_logger.update(
                loss=loss.item(),
                mse_loss=mse_loss.item(),
                penalty_loss=penalty_loss.item(),
                rank_loss=rank_loss.item(),
            )
        if torch.isnan(loss):
            raise RuntimeError("Loss is NaN!")
        if loss < lowest_loss:
            lowest_loss_indices = (lower, upper)
            lowest_loss = loss
        if loss > highest_loss:
            highest_loss_indices = (lower, upper)
            highest_loss = loss

        outputs += [output]

    outputs = torch.cat(outputs)

    if isinstance(criterion, CoxPHLoss):
        outputs = predict_coxph_surv(
            train_outputs,
            train_years_to_event,
            train_events,
            outputs,
            output_dir=output_dir,
            sample=1.0,
        )
    else:
        outputs = torch.exp(outputs)

    c_index = lifelines.utils.concordance_index(
        durations.cpu(), outputs.cpu(), events.cpu()
    )
    # c_index = concordance_index(outputs, durations, events)
    # aucs = auc(surv, durations, events)
    metric_logger.update(c_index=c_index)
    # for key, value in aucs.items():
    #     metric_logger.update(**{key: value})

    logger.info(
        "* c-index {c_index:.3f} loss {losses.global_avg:.3f}".format(
            c_index=c_index,
            losses=metric_logger.loss,
        )
    )
    return (
        {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        lowest_loss_indices,
        highest_loss_indices,
    )


class BilateralMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
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

    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser("Train MLP")
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
    parser.add_argument(
        "--log_images",
        default=False,
        type=utils.bool_flag,
        help="""Whether or not to log example images to Tensorboard""",
    )
    parser.add_argument(
        "--early_stop",
        default=False,
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
        "--loss",
        default="DeepCENTLoss",
        type=str,
        choices=["DeepCENTLoss", "CoxPHLoss"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--lambda_p",
        default=1.0,
        type=float,
        help="Lambda on penalty loss for DeepCENTLoss",
    )
    parser.add_argument(
        "--lambda_r",
        default=0.1,
        type=float,
        help="Lambda on rank loss for DeepCENTLoss",
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
        default=10,
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
        "--sample",
        default=None,
        type=float,
        help="Calculate baseline hazards with SAMPLE subset",
    )
    parser.add_argument(
        "--extraction_batch_size",
        default=1,
        type=int,
        help="Per-GPU batch-size for feature extraction",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Per-GPU batch-size for MLP training",
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
        "--fold", default=None, type=int, help="Fix all folds to fold FOLD."
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
        "--features_dir", default=".", help="Path to checkpoint features"
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
    parser.add_argument("--project", default=None, type=str, help="wandb project")
    args = parser.parse_args()
    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))
    if args.world_size is None:
        args.world_size = len(args.devices)
    args.port = str(utils.get_unused_local_port())
    args.output_dir = utils.prepare_output_dir(args.output_dir, args.autolabel)
    Path(args.features_dir).mkdir(parents=True, exist_ok=True)
    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    in_chans = state_dict.pop("in_chans")
    fit_metadata = state_dict.pop("fit_metadata")
    test_metadata = state_dict.pop("test_metadata")

    logger = utils.setup_logging(args.output_dir, f"train_mlp")

    if not os.path.isfile(os.path.join(args.features_dir, "datasets.pth.tar")):
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

        fit_datasets, test_dataset = get_datasets(
            1.0,
            train_transform,
            val_transform,
            test_metadata,
            fit_metadata,
            args.folds,
            args.seed,
            in_chans,
        )
        torch.save(
            {"fit_datasets": fit_datasets, "test_dataset": test_dataset},
            os.path.join(args.features_dir, "datasets.pth.tar"),
        )
        logger.info(f"Data loaded with {len(test_dataset)} test exams")
        for i, (train_dataset, val_dataset) in enumerate(fit_datasets):
            logger.info(
                f"Fold {i}: {len(train_dataset)} training exams and {len(val_dataset)} val exams"
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
        train_mlp,
        nprocs=len(args.devices),
        args=(result_queue, fold_queue, in_chans, args),
        join=True,
    )
    results = [result_queue.get() for i in range(result_queue.qsize())]
    results = pd.DataFrame(results)
    logger.info(
        "training completed in {:.1f} minutes with mean validation c-index {:.3f} +/ {:.3f}; test c-index {:.3f}".format(
            (time.time() - start) / 60.0,
            results.c_index.mean(),
            results.c_index.std(),
            results[pd.isnull(results.fold)].c_index.iloc[0],
        )
    )

    results.to_pickle(Path(args.output_dir) / "results.pkl")
    with (Path(args.output_dir) / "args.json").open("w") as f:
        f.write(json.dumps(vars(args)) + "\n")

    if args.project is not None:
        wandb.login()
        config = args.config if args.config else args
        run = wandb.init(config=config, project=args.project)
        wandb.log(
            {
                "val_c_index": results[~pd.isnull(results.fold)].c_index.mean(),
                "val_c_index_std": results[~pd.isnull(results.fold)].c_index.std(),
                "test_c_index": results[pd.isnull(results.fold)].c_index,
            }
        )
