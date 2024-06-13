import bisect
import io
import json
import random
from pathlib import Path
import ast
from itertools import chain
from collections import defaultdict
from functools import partial
from glob import glob

import numpy as np
import pyarrow as pa
from PIL import Image
from tqdm import tqdm


def get_table(arrow_file):
    """
    Read an arrow file and return an arrow table.
    """
    return pa.ipc.RecordBatchFileReader(pa.memory_map(f"{arrow_file}", "r")).read_all()


def assert_type(data, dtype, msg=''):
    if not isinstance(data, dtype):
        raise ValueError(f'Expected {msg} type {dtype}, got {type(data)}.')


def ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif isinstance(data, dict):
        data = {k: ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # Assert that all elements in data are python integer, not numpy integer.
        # Because numpy integer cannot be serialized to json.
        data = [int(x) for x in data]
    else:
        raise ValueError(f'Expected data type list, tuple, dict or np.ndarray, got {type(data)}.')
    return data


class ArrowIndexV2(object):
    """
    ArrowIndexV2 is a new version of ArrowIndex.

    Parameters
    ----------
    index_file: str or pathlib.Path
        The path of index file. Either index_file or res_dict should be provided.
    res_dict: dict
        The index dict. Either index_file or res_dict should be provided.
    align: int
        Align the length of indices to be a multiple of align. Generally align should be the batch size * world_size.
    shadow_file_fn: callable or dict
        A callable function to map shadow file path to a new path. If None, the shadow file path will not be
        changed. If a dict is provided, the keys are the shadow names to call the function, and the values are the
        callable functions to map the shadow file path to a new path. If a callable function is provided, the key
        is 'default'.

    Examples
    --------
    >>> index_file = 'data.json'
    >>> indexObj = ArrowIndexV2(index_file)
    >>> pil_image = indexObj.get_image(0)
    >>> text = indexObj.get_attribute(0, column='text_zh')

    """
    def __init__(self, index_file=None, res_dict=None, align=1,
                 shadow_file_fn=None, **kwargs):
        if index_file is not None:
            with open(index_file, 'r') as f:
                res_dict = json.load(f)
        elif res_dict is not None:
            pass
        else:
            raise ValueError(f'Either index_file or res_dict should be provided.')

        self.shadow_file_fn = {}
        if shadow_file_fn is not None:
            if not callable(shadow_file_fn) and not isinstance(shadow_file_fn, dict):
                raise ValueError('shadow_file_fn should be a callable function or a dict.')
            if callable(shadow_file_fn):
                self.shadow_file_fn['default'] = shadow_file_fn
            else:
                for k, v in shadow_file_fn.items():
                    if not callable(v):
                        raise ValueError(f'{k} should be a callable function.')
                    self.shadow_file_fn[k] = v

        self._data = res_dict
        self.data_type = res_dict['data_type']
        self.arrow_files = res_dict['arrow_files']
        self.cum_length = res_dict['cum_length']

        self.group_length = res_dict['group_length']
        error_msg = f'Expected group_length type list, got {type(self.group_length)}.'
        if isinstance(self.group_length, dict):
            raise ValueError(f'{error_msg}\nNote: You may using a multi-resolution index file. '
                             'Try `MultiResolutionBucketIndexV2` instead.')
        elif not isinstance(self.group_length, list):
            raise ValueError(error_msg)

        self.indices = res_dict['indices']
        if 'indices_file' in res_dict:
            self.indices_file = res_dict['indices_file']
            if self.indices_file != '':
                indices_file = Path(index_file).parent / self.indices_file
                if Path(indices_file).exists():
                    self.indices = np.load(indices_file)['x']
                else:
                    raise ValueError(f'This Index file contains an extra file {indices_file} which is missed.')
        else:
            self.indices_file = ''

        if not isinstance(self.indices, list) and not isinstance(self.indices, np.ndarray):
            raise ValueError(f'Expected indices type list or np.ndarray, got {type(self.indices)}.')

        if align > 1:
            if isinstance(self.indices, np.ndarray):
                self.indices = self.indices.tolist()
            self.align(align)

        self.indices = np.asarray(self.indices, int)

        if len(self.arrow_files) != len(self.cum_length):
            raise ValueError(f'Length of arrow_files and cum_length does not match. {len(self.arrow_files)} != {len(self.cum_length)}')
        if len(self.arrow_files) != len(self.group_length):
            raise ValueError(f'Length of arrow_files and group_length does not match. {len(self.arrow_files)} != {len(self.group_length)}')
        if len(self.indices) == 0:
            raise ValueError(f'No indices found in index_dict.')
        if isinstance(self.indices, list) and self.indices[-1] > self.cum_length[-1] - 1:
            raise ValueError(f'Indices exceed cum_length.')

        # Warning:
        #  Ensure that indices are an increasing array. Currently,
        #  no checks are performed due to the potential slowness when dealing with hundreds of millions of data points.

        self.bias = self.cum_length

        self._cur_arrow_file = None
        self._cur_table_map = None
        self._cur_table = None
        self._index_bias = 0
        self.last_index = -1

        self._shadow_cur_arrow_file = {}
        self._shadow_cur_table_map = {}
        self._shadow_cur_table = {}
        self._shadow_index_bias = {}
        self.shadow_last_index = {}
        for k in self.shadow_file_fn.keys():
            self._shadow_cur_arrow_file[k] = None
            self._shadow_cur_table_map[k] = None
            self._shadow_cur_table[k] = None
            self._shadow_index_bias[k] = 0
            self.shadow_last_index[k] = -1

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"""
        ArrowIndexV2(
            data_type        {self.data_type}
            indices_file     {self.indices_file}
            arrow_files      Count={len(self.arrow_files):,}  ({self.arrow_files[0]}, ...)
            cum_length       Count={len(self.cum_length):,}  ({self.cum_length[0]}, ...)
            group_length     Count={len(self.group_length):,}  ({self.group_length[0]}, ...)
            indices          Count={len(self.indices):,}
            example_indices  Count={len(self._data['example_indices']):,}
        )
        """

    def check_exists(self):
        for arrow_file in tqdm(self.arrow_files):
            if not Path(arrow_file).exists():
                print(arrow_file)

    def align(self, align):
        """
        Repeat the index so that the length is a multiple of batch_size * world_size.
        """
        if len(self) % align == 0:
            return

        repeat_num = align - len(self) % align
        if repeat_num >= len(self):
            repeat_n = repeat_num // len(self)
            repeat_times = [repeat_n + 1 for _ in self.indices]
            group_length_new = [ll * (repeat_n + 1) for ll in self.group_length]
            repeat_num -= repeat_n * len(self)
        else:
            repeat_times = [1 for _ in range(repeat_num)]
            group_length_new = [ll for ll in self.group_length]

        for i in range(repeat_num):
            repeat_times[-i - 1] += 1

        repeat_start_idx = len(self) - len(repeat_times)

        group_id = -1
        while group_length_new[group_id] == 0:
            group_id -= 1

        # Allocate the remaining indices that need to be repeated,
        # while also counting how many indices have been checked.
        # If the count reaches the group_length, switch to the next group

        # The reason for paying attention to group_length is that when repeating indices,
        # group_length also needs to be updated synchronously..
        group_acc = 0
        for i in range(repeat_num):
            group_length_new[group_id] += 1
            group_acc += 1
            if group_acc == self.group_length[group_id]:
                group_id -= 1
                while group_length_new[group_id] == 0:
                    group_id -= 1
                group_acc = 0

        temp = []
        for i, value in enumerate(self.indices[repeat_start_idx:]):
            temp.extend([value] * repeat_times[i])

        self.indices = np.concatenate([self.indices[:repeat_start_idx], temp])

        self.group_length = group_length_new

    def shuffle(self, seed=None, fast=False):
        """
        It takes about 30 seconds for an index consisting of 100_000 arrows.
        """
        if fast:
            return self.shuffle_fast(seed)

        indices = self.indices.tolist()

        if seed is not None:
            state = random.getstate()
            random.seed(seed)

        indices_group_list = []
        group_cum_len = 0
        for group_len in self.group_length:
            indices_group = indices[group_cum_len:group_cum_len + group_len]
            random.shuffle(indices_group)
            indices_group_list.append((indices_group, group_len))
            group_cum_len += group_len
        random.shuffle(indices_group_list)
        self.group_length = [x[1] for x in indices_group_list]
        self.indices = np.asarray(list(chain.from_iterable([x[0] for x in indices_group_list])))

        if seed is not None:
            random.setstate(state)

    def shuffle_fast(self, seed=None):
        if seed is not None:
            sampler = np.random.RandomState(seed)
            sampler.shuffle(self.indices)
        else:
            np.random.shuffle(self.indices)

    def get_table(self, arrow_file, shadow=None):
        """
        Read an arrow file and return an arrow table.
        """
        if shadow is None:
            if self._cur_table is not None:
                if self._cur_arrow_file == arrow_file:
                    # This is the same arrow file. Return the cached table.
                    return self._cur_table
                else:
                    # This is a different arrow file. Clear the cache.
                    self._cur_table_map.close()
                    self._cur_table = None

            self._cur_arrow_file = arrow_file
            self._cur_table_map = pa.memory_map(f"{arrow_file}", "r")
            self._cur_table = pa.ipc.RecordBatchFileReader(self._cur_table_map).read_all()
            return self._cur_table
        else:
            if self._shadow_cur_table[shadow] is not None:
                if self._shadow_cur_arrow_file[shadow] == arrow_file:
                    return self._shadow_cur_table[shadow]
                else:
                    self._shadow_cur_table_map[shadow].close()
                    self._shadow_cur_table[shadow] = None

            self._shadow_cur_arrow_file[shadow] = arrow_file
            self._shadow_cur_table_map[shadow] = pa.memory_map(f"{arrow_file}", "r")
            self._shadow_cur_table[shadow] = pa.ipc.RecordBatchFileReader(self._shadow_cur_table_map[shadow]).read_all()
            return self._shadow_cur_table[shadow]

    def get_arrow_file_by_index(self, index, return_index_bias=False, shadow=None):
        i = bisect.bisect_right(self.cum_length, index)
        arrow_file = self.arrow_files[i]

        if return_index_bias:
            if i == 0:
                index_bias = 0
            else:
                index_bias = self.cum_length[i - 1]

            return arrow_file, index_bias

        return arrow_file

    def get_arrow_file(self, ind, shadow=None):
        """
        Get arrow file by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        shadow: str
            The shadow name. If None, return the main arrow file. If not None, return the shadow arrow file.

        Returns
        -------
        arrow_file: str
            The arrow file path.
        """
        index = self.indices[ind]
        return self.get_arrow_file_by_index(index, shadow=shadow)

    def load_table_by_index(self, index, shadow=None):
        if shadow is None:
            if index == self.last_index:
                return self._cur_table
            arrow_file, self._index_bias = \
                self.get_arrow_file_by_index(index, return_index_bias=True)
            self._cur_table = self.get_table(arrow_file)
            self.last_index = index
            return self._cur_table
        else:
            if index == self.shadow_last_index[shadow]:
                return self._shadow_cur_table[shadow]
            shadow_arrow_file, _shadow_index_bias = \
                self.get_arrow_file_by_index(index, return_index_bias=True, shadow=shadow)
            self._shadow_index_bias[shadow] = _shadow_index_bias
            self._shadow_cur_table[shadow] = self.get_table(shadow_arrow_file, shadow=shadow)
            self.shadow_last_index[shadow] = index
            return self._shadow_cur_table[shadow]

    def get_data_by_index(self, index, columns=None, allow_missing=False, return_meta=True, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        if isinstance(columns, str):
            columns = [columns]
        if columns is None:
            columns = list(table.column_names)

        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]
        in_arrow_index = index - index_bias
        if return_meta:
            cur_arrow_file = self._cur_arrow_file if shadow is None else self._shadow_cur_arrow_file[shadow]
            data = {
                'index': index,
                'in_arrow_index': in_arrow_index,
                'arrow_name': cur_arrow_file,
            }
        else:
            data = {}

        if allow_missing:
            for col in columns:
                if col in table.column_names:
                    data[col] = table[col][in_arrow_index].as_py()
        else:
            for col in columns:
                data[col] = table[col][in_arrow_index].as_py()
        return data

    def get_data(self, ind, columns=None, allow_missing=False, return_meta=True, shadow=None):
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
        shadow: str
            The shadow name. If None, return the main data. If not None, return the shadow data.

        Returns
        -------
        data: dict
            A dict containing the data.
        """
        index = self.indices[ind]
        return self.get_data_by_index(index, columns, allow_missing=allow_missing, return_meta=return_meta,
                                      shadow=shadow)

    def get_attribute_by_index(self, index, column, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]
        return table[column][index - index_bias].as_py()

    def get_attribute(self, ind, column, shadow=None):
        """
        Get single attribute by in-dataset index.

        Parameters
        ----------
        ind: int
            The in-dataset index.
        column: str
            The column name.
        shadow: str
            The shadow name. If None, return the main data. If not None, return the shadow data.

        Returns
        -------
        data: can be any type
        """
        index = self.indices[ind]
        return self.get_attribute_by_index(index, column, shadow=shadow)

    def get_image_by_index(self, index, column='image', ret_type='pil', max_size=-1, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]

        col = 'image' if 'image' in table.column_names else 'binary'
        temp = table[col][index - index_bias].as_py()
        image_bytes = io.BytesIO(temp)
        image_bytes.seek(0)
        try:
            # convert(RGB) has two purposes:
            # 1. Convert the image to RGB mode. Some images are in grayscale/RGBA mode, which will cause channel
            #    inconsistency in following processing.
            # 2. Convert the image to RGB mode. Some images are in P mode, which will be forced to use NEAREST resample
            #    method in resize (even if you specify LANCZOS), which will cause blurry images.
            pil_image = Image.open(image_bytes).convert("RGB")
        except Exception as e:
            print(f'get_image_by_index | Error: {e} ({self.get_arrow_file_by_index(index), index - index_bias})')
            pil_image = Image.new("RGB", (256, 256), (255, 255, 255))

        if max_size > 0:
            # Resize the image to max_size. max_size is the size of long edge
            w, h = pil_image.size
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            pil_image = pil_image.resize((new_w, new_h))

        if ret_type == 'numpy':
            return np.array(pil_image)

        return pil_image

    def get_image(self, ind, column='image', ret_type='pil', max_size=-1, shadow=None):
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
        shadow: str
            The shadow name. If None, return the main image. If not None, return the shadow image.

        Returns
        -------
        image: PIL.Image.Image or np.ndarray
        """
        index = self.indices[ind]
        return self.get_image_by_index(index, column, ret_type, max_size, shadow=shadow)

    def get_md5_by_index(self, index, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        index_bias = self._index_bias if shadow is None else self._shadow_index_bias[shadow]
        return table['md5'][index - index_bias].as_py()

    def get_md5(self, ind, shadow=None):
        index = self.indices[ind]
        return self.get_md5_by_index(index, shadow=shadow)

    def get_columns_by_index(self, index, shadow=None):
        table = self.load_table_by_index(index, shadow=shadow)
        return table.column_names

    def get_columns(self, ind, shadow=None):
        index = self.indices[ind]
        return self.get_columns_by_index(index, shadow=shadow)

    def source_distribution(self, save_path=None, shadow=None):
        sources = defaultdict(int)
        for index in tqdm(self.indices):
            source = self.get_attribute_by_index(index, 'source', shadow=shadow)
            sources[source] += 1

        sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)
        for k, v in sources:
            print(f'{k:20s} {v:10d}')
        if save_path is not None:
            Path(save_path).write_text(
                '\n'.join([f'{k:20s} {v:10d}' for k, v in sources]))

    def save(self, save_path):
        """
        Save the index to a json file.

        Parameters
        ----------
        save_path: str or pathlib.Path
            The path to save the index file.
        """
        builder = IndexV2Builder(data_type=self.data_type,
                                 arrow_files=self.arrow_files,
                                 cum_length=self.cum_length,
                                 indices=self.indices,
                                 )
        builder.build(save_path)

    def sample_batch_indices(self, n):
        return np.random.choice(self.indices, n)

    def sample_batch(self, n, columns, progress=True, shadow=None):
        if isinstance(n, int):
            indices = self.sample_batch_indices(n)
        else:
            indices = n

        if progress:
            pbar = tqdm(indices)
        else:
            pbar = indices

        batch_data = []
        for i in pbar:
            batch_data.append(self.get_data_by_index(i, columns, shadow=shadow))
        return batch_data

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
        tw, th = target_size
        w, h = image.size

        tr = th / tw
        r = h / w

        # resize
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        image = image.resize((resize_width, resize_height), resample=resample)

        if crop_type == 'center':
            crop_top = int(round((resize_height - th) / 2.0))
            crop_left = int(round((resize_width - tw) / 2.0))
        elif crop_type == 'random':
            crop_top = random.randint(0, resize_height - th)
            crop_left = random.randint(0, resize_width - tw)
        else:
            raise ValueError(f'crop_type must be center or random, but got {crop_type}')

        image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))
        return image, (crop_left, crop_top)


class IndexV2Builder(object):
    def __init__(self,
                 arrow_files,
                 indices=None,
                 cum_length=None,
                 group_length=None,
                 data_type=None,
                 max_indices=5_000_000,
                 example_num=1000,
                 config_file=None,
                 ):
        """
        Build index v2 from an index dict.

        Parameters
        ----------
        arrow_files: list
            A list of arrow files.
        indices: list or dict
            A list of indices or a dict of indices.
            If not provided, it will be specified as range(cum_length[-1]).
        cum_length: list
            A list of cumulative length of arrow files.
            If not provided, it will be calculated from arrow files.
        group_length: list
            A list of group length or a dict of group length for each arrow file.
            If not provided, it will be calculated.
        data_type: str or list
            Some custom information of this index.
        max_indices: int
            If the number of indices is larger than max_indices, the indices will be saved in a separate file.
            Default to 5_000_000.
        example_num: int
            The number of examples to be saved in the index file. Default to 1000.
        config_file: str
            The path of config file.

        Examples
        --------
        >>> builder = IndexV2Builder(
        >>>     data_type='gold',
        >>>     arrow_files=arrow_files,
        >>>     cum_length=cum_length,
        >>>     indices=indices,
        >>> )
        >>> builder.build(save_path)

        """
        self.arrow_files = arrow_files
        self.indices = indices
        self.cum_length = cum_length
        self.group_length = group_length
        self.data_type = data_type
        self.max_indices = max_indices
        self.example_num = example_num
        self.config_file = config_file

        if isinstance(arrow_files, str):
            if '*' in arrow_files or '?' in arrow_files:
                self.arrow_files = list(glob(arrow_files))
            else:
                self.arrow_files = [arrow_files]
        elif isinstance(self.arrow_files, tuple):
            self.arrow_files = list(self.arrow_files)
        if not isinstance(self.arrow_files, list):
            raise ValueError(f'Expected arrow_files to be a list, got {type(self.arrow_files)}.')

        if self.cum_length is None:
            continuous = False
            if self.indices is None:
                self.group_length = []
                continuous = True

            print(f"Calculating cum_length...")
            self.cum_length = []
            cur_cum_length = 0
            pbar = tqdm(self.arrow_files)
            for arrow_file in pbar:
                table_length = len(get_table(arrow_file))
                cur_cum_length += table_length
                self.cum_length.append(cur_cum_length)
                pbar.set_description(f"{self.cum_length[-1]:>12d}")

                if continuous:
                    self.group_length.append(table_length)

        if self.indices is None:
            self.indices = list(range(self.cum_length[-1]))

        if self.group_length is None:
            self.group_length = []

        if self.data_type is None:
            self.data_type = ['Made by IndexV2Builder']
        elif isinstance(self.data_type, str):
            self.data_type = [self.data_type]

        assert_type(self.data_type, list, 'data_type')
        assert_type(self.cum_length, (list, np.ndarray), 'cum_length')
        assert_type(self.group_length, (list, dict, np.ndarray), 'group_length')
        assert_type(self.indices, (list, dict, np.ndarray), 'indices')
        self.cum_length = ndarray_to_list(self.cum_length)
        self.group_length = ndarray_to_list(self.group_length)
        self.indices = ndarray_to_list(self.indices)

        if isinstance(self.indices, dict):
            for k, v in self.indices.items():
                assert_type(v, list, f'indices[{k}]')

        if len(self.arrow_files) != len(self.cum_length):
            raise ValueError(f'Length of arrow_files and cum_length does not match. {len(self.arrow_files)} != {len(self.cum_length)}')
        if len(self.indices) == 0:
            raise ValueError(f'No indices found in index_dict.')
        if isinstance(self.indices, list) and self.indices[-1] > self.cum_length[-1] - 1:
            raise ValueError(f'Indices exceed cum_length. {self.indices[-1]} > {self.cum_length[-1] - 1}')
        if len(self.group_length) > 0:
            if len(self.arrow_files) != len(self.group_length):
                raise ValueError(f'Length of arrow_files and group_length does not match. {len(self.arrow_files)} != {len(self.group_length)}')
            if sum(self.group_length) != len(self.indices):
                raise ValueError(f'Sum of group_length does not match length of indices. {sum(self.group_length)} != {len(self.indices)}')

    def encode(self):
        # Encode arrow files
        print("Encoding arrow files...")
        arrow_files = []
        for arrow_file in tqdm(self.arrow_files):
            shortname = arrow_file
            arrow_files.append(shortname)
        self.arrow_files = arrow_files

        # Calculate group_length
        print("Calculating group length...")
        if isinstance(self.indices, list):
            if len(self.group_length) == 0:
                self.group_length = self.calc_group_length(self.indices, self.cum_length)
            else:
                print("Group length already calculated, skip.")
        elif isinstance(self.indices, dict):
            if not isinstance(self.group_length, dict):
                self.group_length = {}
            for k, v in self.indices.items():
                print(f"Calculating group length for {k}...")
                if k not in self.group_length or len(self.group_length[k]) == 0:
                    self.group_length[k] = self.calc_group_length(v, self.cum_length)
                else:
                    print("Group length already calculated, skip.")
        else:
            raise ValueError(f'Expected indices type list or dict, got {type(self.indices)}.')

        return {
            'data_type': self.data_type,
            'config_file': self.config_file if self.config_file is not None else '',
            'indices_file': '',
            'arrow_files': self.arrow_files,
            'cum_length': self.cum_length,
            'group_length': self.group_length,
            'indices': self.indices,
            'example_indices': [],
        }

    def build(self, save_path):
        return self.save(save_path)

    def save(self, save_path):
        """
        Make index v2 from an index dict.

        Parameters
        ----------
        save_path: str or pathlib.Path
            The path to save the index file.
        """
        index_dict = self.encode()
        # Ensure the indices either a list or a dict.

        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)

        if isinstance(index_dict['indices'], list) and len(index_dict['indices']) > self.max_indices:
            self.example_indices = index_dict['indices'][:self.example_num]
            indices_to_save = {'x': index_dict['indices']}
            index_dict['indices'] = []
        elif isinstance(index_dict['indices'], dict):
            indices_to_save = index_dict['indices']
            index_dict['indices'] = {}
            num_keys = len(indices_to_save)
            example_num_per_key = max(self.example_num // num_keys, 10)
            index_dict['example_indices'] = {k: v[:example_num_per_key] for k, v in index_dict['indices'].items()}
        else:
            indices_to_save = None

        # save indices
        if indices_to_save is not None:
            indices_file = save_path.parent / f'{save_path.stem}.index'
            indices_dict = {k: np.array(v) for k, v in indices_to_save.items()}
            np.savez_compressed(indices_file, **indices_dict)
            index_dict['indices_file'] = indices_file.name + '.npz'

        with save_path.open('w') as f:
            json.dump(index_dict, f, indent=4, ensure_ascii=False)

    @staticmethod
    def calc_group_length(indices, cum_length):
        group_lengths = []
        cum_ind = 0
        count = 0
        for index in tqdm(indices):
            if index < cum_length[cum_ind]:
                # index is still in the current group
                count += 1
            else:
                # index has exceeded the current group, need to switch to the next group
                group_lengths.append(count)
                cum_ind += 1
                # if the index exceeds the next group, continue to switch to the next group
                while index >= cum_length[cum_ind]:
                    group_lengths.append(0)
                    cum_ind += 1
                count = 1
        # The indices array is exhausted, and the last group containing the index should also be added.
        group_lengths.append(count)
        assert len(group_lengths) <= len(cum_length), (len(group_lengths), len(cum_length))
        # Check if the number of groups is less than the number of cum_length,
        # then the last n groups are empty and need to be filled with zeros.
        if len(group_lengths) < len(cum_length):
            group_lengths.extend([0] * (len(cum_length) - len(group_lengths)))

        return group_lengths