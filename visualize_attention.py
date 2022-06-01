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
import sys
import argparse
import cv2
import random
import colorsys

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
import seaborn as sns

import vision_transformer as vits

from maicara.data.constants import CHIMEC_MEAN, CHIMEC_STD
from chimec import ChiMECSSLDataset, ChiMECStackedSSLDataset
from viz_defaults import *


cmaps = [
    sns.cubehelix_palette(start=s, light=1, as_cmap=True) for s in np.linspace(0, 3, 6)
]
palettes = [sns.cubehelix_palette(start=s, light=1) for s in np.linspace(0, 3, 6)]
# husl = sns.color_palette("husl", 6)


def apply_mask(image, mask, color, alpha=1.0):
    for c in range(3):
        image[:, :, c] = (
            image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
        )
    return image


def display_instances(
    image,
    mask,
    color,
    fname="test",
    figsize=(5, 5),
    blur=False,
    contour=True,
    alpha=0.5,
):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis("off")
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect="auto")
    fig.savefig(fname)
    print(f"{fname} saved.")
    return masked_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--start_index",
        default=0,
        type=int,
        help="Index in dataset to start visualizations",
    )
    parser.add_argument(
        "--image_size", default=(480, 240), type=int, nargs="+", help="Resize image."
    )
    parser.add_argument(
        "--num_examples",
        default=4,
        type=int,
        help="Number of example images to visualize.",
    )
    parser.add_argument(
        "--output_dir",
        default="attention_maps",
        help="Path where to save visualizations.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""",
    )
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
    try:
        in_chans = checkpoint["in_chans"]
    except KeyError:
        in_chans = 1
    # build model
    model = vits.__dict__[args.arch](
        patch_size=args.patch_size, num_classes=0, in_chans=in_chans
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if args.checkpoint_key is not None and args.checkpoint_key in checkpoint:
        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        state_dict = checkpoint[args.checkpoint_key]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            args.pretrained_weights, msg
        )
    )
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(args.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(CHIMEC_MEAN, CHIMEC_STD),
        ]
    )
    SSLDataset = ChiMECStackedSSLDataset if in_chans > 1 else ChiMECSSLDataset
    dataset = SSLDataset(
        transform,
        image_size=224,
    )
    data_iter = iter(dataset)
    for _ in range(args.start_index):
        img = next(data_iter)

    axes = None
    for i in range(args.num_examples):
        print(f"starting example {i}")
        img = next(data_iter)
        # make the image divisible by the patch size
        w, h = (
            img.shape[1] - img.shape[1] % args.patch_size,
            img.shape[2] - img.shape[2] % args.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.get_last_selfattention(img.to(device))

        nh = attentions.shape[1]  # number of heads
        num_columns = (nh + 1) if args.threshold is None else (nh + 2)
        if axes is None:
            fig, axes = plt.subplots(args.num_examples, num_columns, figsize=(20, 15))
            for ax in axes:
                for cell in ax:
                    cell.set_axis_off()

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
                )[0]
                .cpu()
                .numpy()
            )

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(
            torchvision.utils.make_grid(img, normalize=True, scale_each=True),
            os.path.join(args.output_dir, "img.png"),
        )
        axes[i, 0].imshow(img.squeeze(0).squeeze(0), cmap="gray")
        axes[i, 0].axis("off")
        for j in range(nh):
            axes[i, j + 1].imshow(attentions[j], cmap=cmaps[j])
            fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format="png")
            print(f"{fname} saved.")

        if args.threshold is not None:
            image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
            for j in range(nh):
                image = display_instances(
                    image,
                    th_attn[j],
                    palettes[j][2],
                    fname=os.path.join(
                        args.output_dir,
                        f"mask_thr_ex{i}_thr_{args.threshold}_head{j}.png",
                    ),
                    blur=False,
                    contour=False,
                )
            axes[i, nh + 1].imshow(image.astype(np.uint8))
fig.tight_layout()
fig.savefig("attention_maps/attention_examples.png")
