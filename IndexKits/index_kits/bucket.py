import bisect
import json
import random
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union

import numpy as np
from tqdm import tqdm
from PIL import Image

from .indexer import ArrowIndexV2, IndexV2Builder


class Resolution(object):
    def __init__(self, size, *args):
        if isinstance(size, str):
            if 'x' in size:
                size = size.split('x')
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        elif idx == 1:
            return self.w
        else:
            raise IndexError(f'Index {idx} out of range')

    def __str__(self):
        return f'{self.h}x{self.w}'


class ResolutionGroup(object):
    def __init__(self, base_size=None, step=None, align=1, target_ratios=None, enlarge=1, data=None):
        self.enlarge = enlarge

        if data is not None:
            self.data = data
            mid = len(self.data) // 2
            self.base_size = self.data[mid].h
            self.step = self.data[mid].h - self.data[mid - 1].h
        else:
            self.align = align
            self.base_size = base_size
            assert base_size % align == 0, f'base_size {base_size} is not divisible by align {align}'
            if base_size is not None and not isinstance(base_size, int):
                raise ValueError(f'base_size must be None or int, but got {type(base_size)}')
            if step is None and target_ratios is None:
                raise ValueError(f'Either step or target_ratios must be provided')
            if step is not None and step > base_size // 2:
                raise ValueError(f'step must be smaller than base_size // 2, but got {step} > {base_size // 2}')

            self.step = step
            self.data = self.calc(target_ratios)

        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ['' for _ in range(len(self.data))]
        self.prefix_space = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * ' '
        prefix_close = (self.prefix_space - 4) * ' '
        res_str = f'ResolutionGroup(base_size={self.base_size}, step={self.step}, data='
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += f'\n{prefix}ID: height width   ratio {" " * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}'
        res_str += ('\n' + prefix).join([f'{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  '
                                         f'({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}'
                                         for i, x in enumerate(self.data)])
        res_str += f'\n{prefix_close})'
        return res_str

    @staticmethod
    def from_list_of_hxw(hxw_list):
        data = [Resolution(x) for x in hxw_list]
        data = sorted(data, key=lambda x: x.ratio)
        return ResolutionGroup(None, data=data)

    def calc(self, target_ratios=None):
        if target_ratios is None:
            return self._calc_by_step()
        else:
            return self._calc_by_ratio(target_ratios)

    def _calc_by_ratio(self, target_ratios):
        resolutions = []
        for ratio in target_ratios:
            if ratio == '1:1':
                reso = Resolution(self.base_size, self.base_size)
            else:
                hr, wr = map(int, ratio.split(':'))
                x = int((self.base_size ** 2 * self.enlarge // self.align // self.align / (hr * wr)) ** 0.5)
                height = x * hr * self.align
                width = x * wr * self.align
                reso = Resolution(height, width)
            resolutions.append(reso)

        resolutions = sorted(resolutions, key=lambda x_: x_.ratio)

        return resolutions

    def _calc_by_step(self):
        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break

            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(Resolution(cur_height, cur_width))

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break

            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(Resolution(cur_height, cur_width))

        resolutions = sorted(resolutions, key=lambda x: x.ratio)

        return resolutions


class Bucket(ArrowIndexV2):
    def __init__(self, height, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.height = height
        self.width = width
        self.ratio = height / width
        self.scale_dist = []

    def get_scale_by_index(self, index, size_col='hw', shadow=None):
        """
        Calculate the scale to resize the image to fit the bucket.

        Parameters
        ----------
        index: int
            An in-json index.
        size_col: str
            How to get the size of the image. 'hw' for height and width column,
            while 'image' for decoding image binary and get the PIL Image size.
        shadow: str
            The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        scale: float
        """
        if size_col == 'hw':
            w = int(self.get_attribute_by_index(index, 'width', shadow=shadow))
            h = int(self.get_attribute_by_index(index, 'height', shadow=shadow))
        else:
            w, h = self.get_image_by_index(index, shadow=shadow).size
        tw, th = self.width, self.height

        tr = th / tw
        r = h / w

        scale = th / h if r < tr else tw / w
        return scale

    @staticmethod
    def from_bucket_index(index_file, align=1, shadow_file_fn=None):
        with open(index_file, 'r') as f:
            res_dict = json.load(f)

        if not isinstance(res_dict['group_length'], dict):
            error_msg = f'`group_length` must be a dict, but got {type(res_dict["group_length"])}'
            if isinstance(res_dict['group_length'], list):
                raise ValueError(f'{error_msg}\nYou may using a vanilla Index V2 file. Try `ArrowIndexV2` instead.')
            else:
                raise ValueError(error_msg)

        assert 'indices_file' in res_dict, f'indices_file not found in {index_file}'
        assert res_dict['indices_file'] != '', f'indices_file is empty in {index_file}'

        indices_file = Path(index_file).parent / res_dict['indices_file']
        assert Path(indices_file).exists(), f'indices_file {indices_file} not found'

        # Loading indices data
        indices_data = np.load(indices_file)

        # Build buckets
        buckets = []
        keys = []
        for k, v in indices_data.items():
            data = {
                'data_type': res_dict['data_type'],
                'arrow_files': res_dict['arrow_files'],
                'cum_length': res_dict['cum_length'],
            }

            data['indices_file'] = ''
            data['indices'] = v
            data['group_length'] = res_dict['group_length'][k]

            height, width = map(int, k.split('x'))
            bucket = Bucket(height, width, res_dict=data, align=align, shadow_file_fn=shadow_file_fn)

            if len(bucket) > 0:
                buckets.append(bucket)
                keys.append(k)

        resolutions = ResolutionGroup.from_list_of_hxw(keys)
        resolutions.attr = [f'{len(bucket):,d}' for bucket in buckets]

        return buckets, resolutions


class MultiIndexV2(object):
    """
    Multi-bucket index. Support multi-GPU (either single node or multi-node distributed) training.

    Parameters
    ----------
    index_files: list
        The index files.
    batch_size: int
        The batch size of each GPU. Required when using MultiResolutionBucketIndexV2 as base index class.
    world_size: int
        The number of GPUs. Required when using MultiResolutionBucketIndexV2 as base index class.
    sample_strategy: str
        The sample strategy. Can be 'uniform' or 'probability'. Default to 'uniform'.
        If set to probability, a list of probability must be provided. The length of the list must be the same
        as the number of buckets. Each probability value means the sample rate of the corresponding bucket.
    probability: list
        A list of probability. Only used when sample_strategy=='probability'.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.
    seed: int
        Only used when sample_strategy=='probability'. The seed to sample the indices.
    """
    buckets: List[ArrowIndexV2]

    def __init__(self,
                 index_files: List[str],
                 batch_size: Optional[int] = None,
                 world_size: Optional[int] = None,
                 sample_strategy: str = 'uniform',
                 probability: Optional[List[float]] = None,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 seed: Optional[int] = None,
                 ):
        self.buckets = self.load_buckets(index_files,
                                         batch_size=batch_size, world_size=world_size, shadow_file_fn=shadow_file_fn)

        self.sample_strategy = sample_strategy
        self.probability = probability
        self.check_sample_strategy(sample_strategy, probability)

        self.cum_length = self.calc_cum_length()

        self.sampler = np.random.RandomState(seed)
        if sample_strategy == 'uniform':
            self.total_length = sum([len(bucket) for bucket in self.buckets])
            self.ind_mapper = np.arange(self.total_length)
        elif sample_strategy == 'probability':
            self.ind_mapper = self.sample_indices_with_probability()
            self.total_length = len(self.ind_mapper)
        else:
            raise ValueError(f"Not supported sample_strategy {sample_strategy}.")

    def load_buckets(self, index_files, **kwargs):
        buckets = [ArrowIndexV2(index_file, **kwargs) for index_file in index_files]
        return buckets

    def __len__(self):
        return self.total_length

    def check_sample_strategy(self, sample_strategy, probability):
        if sample_strategy == 'uniform':
            pass
        elif sample_strategy == 'probability':
            if probability is None:
                raise ValueError(f"probability must be provided when sample_strategy is 'probability'.")
            assert isinstance(probability, (list, tuple)), \
                f"probability must be a list, but got {type(probability)}"
            assert len(self.buckets) == len(probability), \
                f"Length of index_files {len(self.buckets)} != Length of probability {len(probability)}"
        else:
            raise ValueError(f"Not supported sample_strategy {sample_strategy}.")

    def sample_indices_with_probability(self):
        ind_mapper_list = []
        accu = 0
        for bucket, p in zip(self.buckets, self.probability):
            if p == 1:
                # Just use all indices
                indices = np.arange(len(bucket)) + accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                indices_part1 = np.arange(len(bucket)).repeat(repeat_times)
                indices_part2 = self.sampler.choice(len(bucket), int(len(bucket) * (p - repeat_times)), replace=False)
                indices = np.sort(np.concatenate([indices_part1, indices_part2])) + accu
            ind_mapper_list.append(indices)
            accu += len(bucket)
        ind_mapper = np.concatenate(ind_mapper_list)
        return ind_mapper

    def calc_cum_length(self):
        cum_length = []
        length = 0
        for bucket in self.buckets:
            length += len(bucket)
            cum_length.append(length)
        return cum_length

    def shuffle(self, seed=None, fast=False):
        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indexes
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i, fast=fast)

        # Shuffle ind_mapper
        if self.sample_strategy == 'uniform':
            self.ind_mapper = np.arange(self.total_length)
        elif self.sample_strategy == 'probability':
            self.ind_mapper = self.sample_indices_with_probability()
        else:
            raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(self.ind_mapper)
        else:
            np.random.shuffle(self.ind_mapper)

    def get_arrow_file(self, ind, **kwargs):
        """
        Get arrow file by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        arrow_file: str
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_arrow_file(ind - bias, **kwargs)

    def get_data(self, ind, columns=None, allow_missing=False, return_meta=True, **kwargs):
        """
        Get data by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        columns: str or list
            The columns to be returned. If None, return all columns.
        allow_missing: bool
            If True, omit missing columns. If False, raise an error if the column is missing.
        return_meta: bool
            If True, the resulting dict will contain some meta information:
            in-json index, in-arrow index, and arrow_name.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        data: dict
            A dict containing the data.
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_data(ind - bias, columns=columns, allow_missing=allow_missing,
                                        return_meta=return_meta, **kwargs)

    def get_attribute(self, ind, column, **kwargs):
        """
        Get single attribute by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            The column name.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        attribute: Any
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_attribute(ind - bias, column, **kwargs)

    def get_image(self, ind, column='image', ret_type='pil', max_size=-1, **kwargs):
        """
        Get image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            [Deprecated] The column name of the image. Default to 'image'.
        ret_type: str
            The return type. Can be 'pil' or 'numpy'. Default to 'pil'.
        max_size: int
            If not -1, resize the image to max_size. max_size is the size of long edge.
        kwargs: dict
            shadow: str
                The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        image: PIL.Image.Image or np.ndarray
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_image(ind - bias, column, ret_type, max_size, **kwargs)

    def get_md5(self, ind, **kwargs):
        """ Get md5 by in-dataset index. """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_md5(ind - bias, **kwargs)

    def get_columns(self, ind, **kwargs):
        """ Get columns by in-dataset index. """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_columns(ind - bias, **kwargs)

    @staticmethod
    def resize_and_crop(image, target_size, resample=Image.LANCZOS, crop_type='random'):
        """
        Resize image without changing aspect ratio, then crop the center/random part.

        Parameters
        ----------
        image: PIL.Image.Image
            The input image to be resized and cropped.
        target_size: tuple
            The target size of the image.
        resample:
            The resample method. See PIL.Image.Image.resize for details. Default to Image.LANCZOS.
        crop_type: str
            'center' or 'random'. If 'center', crop the center part of the image. If 'random',
            crop a random part of the image. Default to 'random'.

        Returns
        -------
        image: PIL.Image.Image
            The resized and cropped image.
        crop_pos: tuple
            The position of the cropped part. (crop_left, crop_top)
        """
        return ArrowIndexV2.resize_and_crop(image, target_size, resample, crop_type)


class MultiResolutionBucketIndexV2(MultiIndexV2):
    """
    Multi-resolution bucket index. Support multi-GPU (either single node or multi-node distributed) training.

    Parameters
    ----------
    index_file: str
        The index file of the bucket index.
    batch_size: int
        The batch size of each GPU.
    world_size: int
        The number of GPUs.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.
    """
    buckets: List[Bucket]

    def __init__(self,
                 index_file: str,
                 batch_size: int,
                 world_size: int,
                 shadow_file_fn: Optional[Union[Callable, Dict[str, Callable]]] = None,
                 ):
        align = batch_size * world_size
        if align <= 0:
            raise ValueError(f'Align size must be positive, but got {align} = {batch_size} x {world_size}')

        self.buckets, self._resolutions = Bucket.from_bucket_index(index_file,
                                                                   align=align,
                                                                   shadow_file_fn=shadow_file_fn,
                                                                   )
        self.arrow_files = self.buckets[0].arrow_files
        self._base_size = self._resolutions.base_size
        self._step = self._resolutions.step

        self.buckets = sorted(self.buckets, key=lambda x: x.ratio)
        self.cum_length = self.calc_cum_length()

        self.total_length = sum([len(bucket) for bucket in self.buckets])
        assert self.total_length % align == 0, f'Total length {self.total_length} is not divisible by align size {align}'

        self.align_size = align
        self.batch_size = batch_size
        self.world_size = world_size
        self.ind_mapper = np.arange(self.total_length)

    @property
    def step(self):
        return self._step

    @property
    def base_size(self):
        return self._base_size

    @property
    def resolutions(self):
        return self._resolutions

    def shuffle(self, seed=None, fast=False):
        # Shuffle indexes
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i, fast=fast)

        # Shuffle ind_mapper
        batch_ind_mapper = np.arange(self.total_length // self.batch_size) * self.batch_size
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(batch_ind_mapper)
        else:
            np.random.shuffle(batch_ind_mapper)
        ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)
        self.ind_mapper = ind_mapper

    def get_ratio(self, ind, **kwargs):
        """
        Get the ratio of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        width, height, ratio
        """
        ind = self.ind_mapper[ind]
        width, height = self.get_image(ind, **kwargs).size
        return width, height, height / width

    def get_target_size(self, ind):
        """
        Get the target size of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        target_width, target_height
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        return self.buckets[i].width, self.buckets[i].height

    def scale_distribution(self, save_file=None):
        if save_file is not None:
            scale_dict = np.load(save_file)
            for bucket in self.buckets:
                bucket.scale_dist = scale_dict[f'{bucket.height}x{bucket.width}']
        else:
            for bucket in tqdm(self.buckets):
                for index in tqdm(bucket.indices, leave=False):
                    scale = bucket.get_scale_by_index(index)
                    bucket.scale_dist.append(scale)
            scale_dict = {f'{bucket.height}x{bucket.width}': bucket.scale_dist for bucket in self.buckets}

            if save_file is not None:
                save_file = Path(save_file)
                save_file.parent.mkdir(exist_ok=True, parents=True)
                np.savez_compressed(save_file, **scale_dict)

        return self


class MultiMultiResolutionBucketIndexV2(MultiIndexV2):
    buckets: List[MultiResolutionBucketIndexV2]

    @property
    def step(self):
        return [b.step for b in self.buckets]

    @property
    def base_size(self):
        return [b.base_size for b in self.buckets]

    @property
    def resolutions(self):
        return [b.resolutions for b in self.buckets]

    def load_buckets(self, index_files, **kwargs):
        self.batch_size = kwargs.get('batch_size', None)
        self.world_size = kwargs.get('world_size', None)
        if self.batch_size is None or self.world_size is None:
            raise ValueError("`batch_size` and `world_size` must be provided when using "
                             "`MultiMultiResolutionBucketIndexV2`.")
        buckets = [
            MultiResolutionBucketIndexV2(index_file,
                                         self.batch_size,
                                         self.world_size,
                                         shadow_file_fn=kwargs.get('shadow_file_fn', None),
                                         )
            for index_file in index_files
        ]
        return buckets

    def sample_indices_with_probability(self, return_batch_indices=False):
        bs = self.batch_size
        ind_mapper_list = []
        accu = 0
        for bucket, p in zip(self.buckets, self.probability):
            if p == 1:
                # Just use all indices
                batch_indices = np.arange(len(bucket) // bs) * bs + accu
            else:
                # Use all indices multiple times, and then sample some indices without replacement
                repeat_times = int(p)
                indices_part1 = np.arange(len(bucket) // bs).repeat(repeat_times) * bs
                indices_part2 = self.sampler.choice(len(bucket) // bs, int(len(bucket) * (p / bs - repeat_times)),
                                                    replace=False) * bs
                batch_indices = np.sort(np.concatenate([indices_part1, indices_part2])) + accu

            if return_batch_indices:
                indices = batch_indices
            else:
                indices = np.stack([batch_indices + i for i in range(bs)], axis=1).reshape(-1)
            ind_mapper_list.append(indices)
            accu += len(bucket)
        ind_mapper = np.concatenate(ind_mapper_list)
        return ind_mapper

    def shuffle(self, seed=None, fast=False):
        if self.sample_strategy == 'probability':
            # Notice: In order to resample indices when shuffling, shuffle will not preserve the
            # initial sampled indices when loading the index.
            pass

        # Shuffle indexes
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
            random.shuffle(self.buckets)
            random.setstate(state)
        else:
            random.shuffle(self.buckets)

        self.cum_length = self.calc_cum_length()

        # Shuffle indices in each index
        for i, bucket in enumerate(self.buckets):
            bucket.shuffle(seed + i, fast=fast)

        # Shuffle ind_mapper in batch level
        if self.sample_strategy == 'uniform':
            batch_ind_mapper = np.arange(self.total_length // self.batch_size) * self.batch_size
        elif self.sample_strategy == 'probability':
            batch_ind_mapper = self.sample_indices_with_probability(return_batch_indices=True)
        else:
            raise ValueError(f"Not supported sample_strategy {self.sample_strategy}.")
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(batch_ind_mapper)
        else:
            np.random.shuffle(batch_ind_mapper)
        self.ind_mapper = np.stack([batch_ind_mapper + i for i in range(self.batch_size)], axis=1).reshape(-1)

    def get_ratio(self, ind, **kwargs):
        """
        Get the ratio of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        width, height, ratio
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_ratio(ind - bias, **kwargs)

    def get_target_size(self, ind):
        """
        Get the target size of the image by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.

        Returns
        -------
        target_width, target_height
        """
        ind = self.ind_mapper[ind]
        i = bisect.bisect_right(self.cum_length, ind)
        bias = self.cum_length[i - 1] if i > 0 else 0
        return self.buckets[i].get_target_size(ind - bias)


def build_multi_resolution_bucket(config_file,
                                  base_size,
                                  src_index_files,
                                  save_file,
                                  reso_step=64,
                                  target_ratios=None,
                                  align=1,
                                  min_size=0,
                                  md5_hw=None,
                                  ):
    # Compute base size
    resolutions = ResolutionGroup(base_size, step=reso_step, target_ratios=target_ratios, align=align)
    print(resolutions)

    save_file = Path(save_file)
    save_file.parent.mkdir(exist_ok=True, parents=True)

    if isinstance(src_index_files, str):
        src_index_files = [src_index_files]
    src_indexes = []
    print(f'Loading indexes:')
    for src_index_file in src_index_files:
        src_indexes.append(ArrowIndexV2(src_index_file))
        print(f'    {src_index_file} | cum_length: {src_indexes[-1].cum_length[-1]} | indices: {len(src_indexes[-1])}')

    if md5_hw is None:
        md5_hw = {}

    arrow_files = src_indexes[0].arrow_files[:]     # !!!important!!!, copy the list
    for src_index in src_indexes[1:]:
        arrow_files.extend(src_index.arrow_files[:])

    cum_length = src_indexes[0].cum_length[:]
    for src_index in src_indexes[1:]:
        cum_length.extend([x + cum_length[-1] for x in src_index.cum_length])
    print(f'cum_length: {cum_length[-1]}')

    group_length_list = src_indexes[0].group_length[:]
    for src_index in src_indexes[1:]:
        group_length_list.extend(src_index.group_length[:])

    total_indices = sum([len(src_index) for src_index in src_indexes])
    total_group_length = sum(group_length_list)
    assert total_indices == total_group_length, f'Total indices {total_indices} != Total group length {total_group_length}'

    buckets = [[] for _ in range(len(resolutions))]
    cum_length_tmp = 0
    total_index_count = 0
    for src_index, src_index_file in zip(src_indexes, src_index_files):
        index_count = 0
        pbar = tqdm(src_index.indices.tolist())
        for i in pbar:
            try:
                height = int(src_index.get_attribute_by_index(i, 'height'))
                width = int(src_index.get_attribute_by_index(i, 'width'))
            except Exception as e1:
                try:
                    md5 = src_index.get_attribute_by_index(i, 'md5')
                    height, width = md5_hw[md5]
                except Exception as e2:
                    try:
                        width, height = src_index.get_image_by_index(i).size
                    except Exception as e3:
                        print(f'Error: {e1} --> {e2} --> {e3}. We will skip this image.')
                        continue

            if height < min_size or width < min_size:
                continue

            ratio = height / width
            idx = np.argmin(np.abs(resolutions.ratio - ratio))
            buckets[idx].append(i + cum_length_tmp)
            index_count += 1
        print(f"Valid indices {index_count} in {src_index_file}.")
        cum_length_tmp += src_index.cum_length[-1]
        total_index_count += index_count
    print(f'Total indices: {total_index_count}')

    print(f'Making bucket index.')
    indices = {}
    for i, bucket in tqdm(enumerate(buckets)):
        if len(bucket) == 0:
            continue
        reso = f'{resolutions[i]}'
        resolutions.attr[i] = f'{len(bucket):>6d}'
        indices[reso] = bucket

    builder = IndexV2Builder(data_type=['multi-resolution-bucket-v2',
                                        f'base_size={base_size}',
                                        f'reso_step={reso_step}',
                                        f'target_ratios={target_ratios}',
                                        f'align={align}',
                                        f'min_size={min_size}',
                                        f'src_files='] +
                                       [f'{src_index_file}' for src_index_file in src_index_files],
                             arrow_files=arrow_files,
                             cum_length=cum_length,
                             indices=indices,
                             config_file=config_file,
                             )
    builder.build(save_file)
    print(resolutions)
    print(f'Build index finished!\n\n'
          f'            Save path: {Path(save_file).absolute()}\n'
          f'    Number of indices: {sum([len(v) for k, v in indices.items()])}\n'
          f'Number of arrow files: {len(arrow_files)}\n'
          )
