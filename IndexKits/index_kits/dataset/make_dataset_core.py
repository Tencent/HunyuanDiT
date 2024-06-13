import json
import pickle
from collections import defaultdict
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yaml

import numpy as np
import pyarrow as pa
from tqdm import tqdm

from index_kits.indexer import IndexV2Builder
from index_kits.bucket import build_multi_resolution_bucket
from index_kits.dataset.config_parse import DatasetConfig


def get_table(arrow_file):
    return pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, 'r')).read_all()


def get_indices(arrow_file, repeat_times, filter_fn, repeat_fn, callback=None):
    """
    Get valid indices from a single arrow_file.

    Parameters
    ----------
    arrow_file: str
    repeat_times: int
        Repeat remain indices multiple times.
    filter_fn
    callback

    Returns
    -------

    """
    try:
        table = pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, 'r')).read_all()
    except Exception as e:
        print(arrow_file, e)
        raise e
    length = len(table)

    if len(table) == 0:
        print(f"Warning: Empty table: {arrow_file}")
        indices = []
        stats = {}

    else:
        # Apply filter_fn if available
        if filter_fn is not None:
            mask, stats, md5s = filter_fn(arrow_file, table)
        else:
            mask = pd.Series([True] * length)
            stats = {}
            md5s = None

        # Apply callback function if available
        if callback is not None:
            mask, stats = callback(arrow_file, table, mask, stats, md5s)

        # Get indices
        if mask is not None:
            indices = np.where(mask)[0].tolist()
        else:
            indices = list(range(length))

        # Apply indices repeat
        if repeat_fn is not None:
            indices, repeat_stats = repeat_fn(arrow_file, table, indices, repeat_times, md5s)
            stats.update(repeat_stats)

    return arrow_file, length, indices, stats


def load_md5_files(files, name=None):
    if isinstance(files, str):
        files = [files]
    md5s = set()
    for file in files:
        md5s.update(Path(file).read_text().splitlines())
    print(f"    {name} md5s: {len(md5s):,}")

    return md5s

def load_md52cls_files(files, name=None):
    if isinstance(files, str):
        files = [files]
    md52cls = {}
    for file in files:
        with Path(file).open() as f:
            md52cls.update(json.load(f))
    print(f"    {name} md52cls: {len(md52cls):,}")

    return md52cls


def merge_and_build_index(data_type, src, dconfig, save_path):
    if isinstance(src, str):
        files = list(sorted(glob(src)))
    else:
        files = list(sorted(src))
    print(f"Found {len(files):,} temp pickle files.")
    for fname in files:
        print(f"    {fname}")
    arrow_files = []
    table_lengths = []
    indices_list = []
    bad_stats_total = defaultdict(int)
    total_indices = 0
    total_processed_length = 0
    for file_name in tqdm(files):
        with Path(file_name).open('rb') as f:
            data = pickle.load(f)
        for arrow_file, table_length, indices, *args in tqdm(data, leave=False):
            arrow_files.append(arrow_file)
            table_lengths.append(table_length)
            total_processed_length += table_length
            indices_list.append(indices)
            total_indices += len(indices)
            if len(args) > 0 and args[0]:
                bad_stats = args[0]
                for k, v in bad_stats.items():
                    bad_stats_total[k] += v

    if len(bad_stats_total):
        stats_save_dir = Path(save_path).parent
        stats_save_dir.mkdir(parents=True, exist_ok=True)
        stats_save_path = stats_save_dir / (Path(save_path).stem + '_stats.txt')
        stats_save_path.write_text('\n'.join([f'{k:>50s} {v}' for k, v in bad_stats_total.items()]) + '\n')
        print(f"Save stats to {stats_save_path}")

    print(f'Arrow files: {len(arrow_files):,}')
    print(f'Processed indices: {total_processed_length:,}')
    print(f'Valid indices: {total_indices:,}')

    cum_length = 0
    total_indices = []
    cum_lengths = []
    group_lengths = []
    existed = set()
    print(f"Accumulating indices...")
    pbar = tqdm(zip(arrow_files, table_lengths, indices_list), total=len(arrow_files), mininterval=1)
    _count = 0
    for arrow_file, table_length, indices in pbar:
        if len(indices) > 0 and dconfig.remove_md5_dup:
            new_indices = []
            table = get_table(arrow_file)
            if 'md5' not in table.column_names:
                raise ValueError(f"Column 'md5' not found in {arrow_file}. "
                                 f"When `remove_md5_dup: true` is set, md5 column is required.")
            md5s = table['md5'].to_pandas()
            for i in indices:
                md5 = md5s[i]
                if md5 in existed:
                    continue
                existed.add(md5)
                new_indices.append(i)
            indices = new_indices

        total_indices.extend([int(i + cum_length) for i in indices])
        cum_length += table_length
        cum_lengths.append(cum_length)
        group_lengths.append(len(indices))

        _count += 1

        if _count % 100 == 0:
            pbar.set_description(f'Indices: {len(total_indices):,}')

    builder = IndexV2Builder(data_type=data_type,
                             arrow_files=arrow_files,
                             cum_length=cum_lengths,
                             group_length=group_lengths,
                             indices=total_indices,
                             config_file=dconfig.config_file,
                             )
    builder.build(save_path)
    print(f'Build index finished!\n\n'
          f'            Save path: {Path(save_path).absolute()}\n'
          f'    Number of indices: {len(total_indices)}\n'
          f'Number of arrow files: {len(arrow_files)}\n'
          )


def worker_startup(rank, world_size, dconfig, prefix, work_dir, callback=None):
    # Prepare names for this worker
    num = (len(dconfig.names) + world_size - 1) // world_size
    arrow_names = dconfig.names[rank * num:(rank + 1) * num]
    print(f'Rank {rank} has {len(arrow_names):,} names.')

    # Run get indices
    print(f"Start getting indices...")
    indices = []
    for arrow_name, repeat_times in tqdm(arrow_names, position=rank, desc=f"#{rank}: ", leave=False):
        indices.append(get_indices(arrow_name, repeat_times, dconfig.filter, dconfig.repeater, callback))

    # Save to a temp file
    temp_save_path = work_dir / f'data/temp_pickles/{prefix}-{rank + 1}_of_{world_size}.pkl'
    temp_save_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_save_path.open('wb') as f:
        pickle.dump(indices, f)
    print(f'Rank {rank} finished. Write temporary data to {temp_save_path}')

    return temp_save_path


def startup(config_file,
            save,
            world_size=1,
            work_dir='.',
            callback=None,
            use_cache=False,
            ):
    work_dir = Path(work_dir)
    save_path = Path(save)
    if save_path.suffix != '.json':
        save_path = save_path.parent / (save_path.name + '.json')
    print(f"Using save_path: {save_path}")
    prefix = f"{save_path.stem}"

    # Parse dataset config and build the data_type list
    dconfig = DatasetConfig(work_dir, config_file)
    data_type = []
    for k, v in dconfig.data_type.items():
        data_type.extend(v)
        print(f"{k}:")
        for x in v:
            print(f'    {x}')
    if dconfig.remove_md5_dup:
        data_type.append('Remove md5 duplicates.')
    else:
        data_type.append('Keep md5 duplicates.')

    # Start processing
    if not use_cache:
        temp_pickles = []
        if world_size == 1:
            print(f"\nRunning in single process mode...")
            temp_pickles.append(worker_startup(rank=0,
                                               world_size=1,
                                               dconfig=dconfig,
                                               prefix=prefix,
                                               work_dir=work_dir,
                                               callback=callback,
                                               ))
        else:
            print(f"\nRunning in multi-process mode (world_size={world_size})...")
            p = Pool(world_size)
            temp_pickles_ = []
            for i in range(world_size):
                temp_pickles_.append(p.apply_async(worker_startup, args=(i, world_size, dconfig, prefix, work_dir, callback)))

            for res in temp_pickles_:
                temp_pickles.append(res.get())
            # close
            p.close()
            p.join()
    else:
        temp_pickles = glob(f'{work_dir}/data/temp_pickles/{prefix}-*_of_{world_size}.pkl')

    # Merge temp pickles and build index
    merge_and_build_index(data_type,
                          temp_pickles,
                          dconfig,
                          save_path,
                          )


def make_multireso(target,
                   config_file=None,
                   src=None,
                   base_size=None,
                   reso_step=None,
                   target_ratios=None,
                   align=None,
                   min_size=None,
                   md5_file=None,
                   ):
    if config_file is not None:
        with Path(config_file).open() as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    src = config.get('src', src)
    base_size = config.get('base_size', base_size)
    reso_step = config.get('reso_step', reso_step)
    target_ratios = config.get('target_ratios', target_ratios)
    align = config.get('align', align)
    min_size = config.get('min_size', min_size)
    md5_file = config.get('md5_file', md5_file)

    if src is None:
        raise ValueError('src must be provided in either config file or command line.')
    if base_size is None:
        raise ValueError('base_size must be provided.')
    if reso_step is None and target_ratios is None:
        raise ValueError('Either reso_step or target_ratios must be provided.')

    if md5_file is not None:
        with open(md5_file, 'rb') as f:
            md5_hw = pickle.load(f)
        print(f'Md5 to height and width: {len(md5_hw):,}')
    else:
        md5_hw = None

    build_multi_resolution_bucket(
        config_file=config_file,
        base_size=base_size,
        reso_step=reso_step,
        target_ratios=target_ratios,
        align=align,
        min_size=min_size,
        src_index_files=src,
        save_file=target,
        md5_hw=md5_hw,
    )
