import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.utils import iter_patch

from monai.transforms.utils import convert_pad_mode
from monai.utils import (
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    convert_to_dst_type,
)
from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils.enums import PytorchPadMode, TransformBackends
from monai.utils.type_conversion import convert_data_type


class GridTile(Transform):
    """
    Extract all the tiles sweeping the entire image in a row-major sliding-window manner with possible overlaps.
    It can sort the tiles and return all or a subset of them.

    Adapted from MONAI from pathology-centric focus (fixed number of tiles with sorting) to radiology-centric focus (all non-background tiles).
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/spatial/array.py

    Args:
        tile_size: size of tiles to generate slices for, 0 or None selects whole dimension
        offset: offset of starting position in the array, default is 0 for each dimension.
        overlap: the amount of overlap of neighboring tiles in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        threshold: a value to keep only the tiles whose average intensity is less than the threshold.
            Defaults to no filtering.
        pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to ``"constant"``.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        tile_size: Sequence[int],
        offset: Optional[Sequence[int]] = None,
        overlap: Union[Sequence[float], float] = 0.0,
        max_frac_black: Optional[float] = None,
        pad_mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        self.tile_size = ensure_tuple(tile_size)
        self.offset = ensure_tuple(offset) if offset else (0,) * len(self.tile_size)
        self.pad_mode: Optional[NumpyPadMode] = (
            convert_pad_mode(dst=np.zeros(1), mode=pad_mode) if pad_mode else None
        )
        self.pad_kwargs = pad_kwargs
        self.overlap = overlap
        self.max_frac_black = max_frac_black

    def filter_threshold(self, image_np: np.ndarray, locations: np.ndarray):
        """
        Filter the tiles and their locations according to max_frac_black
        Args:
            image_np: a numpy.ndarray representing a stack of tiles (n_tiles, color_chan, h, w)
            locations: a numpy.ndarray representing the stack of location of each tile
        """
        if self.max_frac_black is not None:
            num_pixels = image_np.shape[2] * image_np.shape[3]
            passes = [
                (
                    p[p == image_np.min()].ravel().shape[0] / num_pixels
                    < self.max_frac_black
                )
                for p in image_np
            ]

            image_np = image_np[passes]
            locations = locations[passes]
        return image_np, locations

    def __call__(self, array: NdarrayOrTensor):
        # create the tile iterator which sweeps the image row-by-row
        array_np, *_ = convert_data_type(array, np.ndarray)
        tile_iterator = iter_patch(
            array_np,
            patch_size=(None,) + self.tile_size,  # expand to have the channel dim
            start_pos=(0,) + self.offset,  # expand to have the channel dim
            overlap=self.overlap,
            copy_back=False,
            mode=self.pad_mode,
            **self.pad_kwargs,
        )
        tiles = list(zip(*tile_iterator))
        tiled_image = np.array(tiles[0])
        locations = np.array(tiles[1])[:, 1:, 0]  # only keep the starting location

        # Filter tiles
        if self.max_frac_black:
            tiled_image, locations = self.filter_threshold(tiled_image, locations)

        # return torch.Tensor(tiled_image), torch.Tensor(locations)
        return torch.Tensor(tiled_image)


class RandGridTile(GridTile, RandomizableTransform):
    """
    Extract all the tiles sweeping the entire image in a row-major sliding-window manner with possible overlaps,
    and with random offset for the minimal corner of the image, (0,0) for 2D and (0,0,0) for 3D.
    It can sort the patches and return all or a subset of them.

    Adapted from MONAI from pathology-centric focus (fixed number of tiles with sorting) to radiology-centric focus (all non-background tiles).
    https://github.com/Project-MONAI/MONAI/blob/dev/monai/transforms/spatial/array.py

    Args:
        tile_size: size of tiles to generate slices for, 0 or None selects whole dimension
        min_offset: the minimum range of offset to be selected randomly. Defaults to 0.
        max_offset: the maximum range of offset to be selected randomly.
            Defaults to image size modulo tile size.
        overlap: the amount of overlap of neighboring tiles in each dimension (a value between 0.0 and 1.0).
            If only one float number is given, it will be applied to all dimensions. Defaults to 0.0.
        pad_mode: refer to NumpyPadMode and PytorchPadMode. If None, no padding will be applied. Defaults to ``"constant"``.
        pad_kwargs: other arguments for the `np.pad` or `torch.pad` function.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        tile_size: Sequence[int],
        min_offset: Optional[Union[Sequence[int], int]] = None,
        max_offset: Optional[Union[Sequence[int], int]] = None,
        max_frac_black: Optional[float] = None,
        pad_mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ):
        super().__init__(
            tile_size=tile_size,
            offset=(),
            overlap=0.0,
            max_frac_black=max_frac_black,
            pad_mode=pad_mode,
            **pad_kwargs,
        )
        self.min_offset = min_offset
        self.max_offset = max_offset

    def randomize(self, array):
        if self.min_offset is None:
            min_offset = (0,) * len(self.tile_size)
        else:
            min_offset = ensure_tuple_rep(self.min_offset, len(self.tile_size))
        if self.max_offset is None:
            max_offset = tuple(s % p for s, p in zip(array.shape[1:], self.tile_size))
        else:
            max_offset = ensure_tuple_rep(self.max_offset, len(self.tile_size))

        self.offset = tuple(
            self.R.randint(low=low, high=high + 1)
            for low, high in zip(min_offset, max_offset)
        )

    def __call__(self, array: NdarrayOrTensor, randomize: bool = True):
        if randomize:
            self.randomize(array)
        return super().__call__(array)
