import json
import numpy as np
from pathlib import Path
from functools import partial

from .indexer import ArrowIndexV2
from .bucket import (
    ResolutionGroup,
    MultiIndexV2, MultiResolutionBucketIndexV2, MultiMultiResolutionBucketIndexV2, IndexV2Builder
)


def load_index(src,
               multireso=False,
               batch_size=1,
               world_size=1,
               sample_strategy='uniform',
               probability=None,
               shadow_file_fn=None,
               seed=None,
               ):
    if isinstance(src, str):
        src = [src]
    if src[0].endswith('.arrow'):
        if multireso:
            raise ValueError('Arrow file does not support multiresolution. Please make base index V2 first and then'
                             'build multiresolution index.')
        idx = IndexV2Builder(src).to_index_v2()
    elif src[0].endswith('.json'):
        if multireso:
            if len(src) == 1:
                idx = MultiResolutionBucketIndexV2(src[0], batch_size=batch_size,
                                                   world_size=world_size,
                                                   shadow_file_fn=shadow_file_fn,
                                                   )
            else:
                idx = MultiMultiResolutionBucketIndexV2(src, batch_size=batch_size,
                                                        world_size=world_size,
                                                        sample_strategy=sample_strategy, probability=probability,
                                                        shadow_file_fn=shadow_file_fn, seed=seed,
                                                        )
        else:
            if len(src) == 1:
                idx = ArrowIndexV2(src[0],
                                   shadow_file_fn=shadow_file_fn,
                                   )
            else:
                idx = MultiIndexV2(src,
                                   sample_strategy=sample_strategy, probability=probability,
                                   shadow_file_fn=shadow_file_fn, seed=seed,
                                   )
    else:
        raise ValueError(f'Unknown file type: {src[0]}')
    return idx


def get_attribute(data, attr_list):
    ret_data = {}
    for attr in attr_list:
        ret_data[attr] = data.get(attr, None)
        if ret_data[attr] is None:
            raise ValueError(f'Missing {attr} in {data}')
    return ret_data


def get_optional_attribute(data, attr_list):
    ret_data = {}
    for attr in attr_list:
        ret_data[attr] = data.get(attr, None)
    return ret_data


def detect_index_type(data):
    if isinstance(data['group_length'], dict):
        return 'multireso'
    else:
        return 'base'

def show_index_info(src, only_arrow_files=False, depth=1):
    """
    Show base/multireso index information.
    """
    if not Path(src).exists():
        raise ValueError(f'{src} does not exist.')
    print(f"Loading index file {src} ...")
    with open(src, 'r') as f:
        src_data = json.load(f)
    print(f"Loaded.")
    data = get_attribute(src_data, ['data_type', 'indices_file', 'arrow_files', 'cum_length',
                                'group_length', 'indices', 'example_indices'])
    opt_data = get_optional_attribute(src_data, ['config_file'])

    # Format arrow_files examples
    arrow_files = data['arrow_files']
    if only_arrow_files:
        existed = set()
        arrow_files_output_list = []
        for arrow_file in arrow_files:
            if depth == 0:
                if arrow_file not in existed:
                    arrow_files_output_list.append(arrow_file)
                    existed.add(arrow_file)
            elif depth > 0:
                parts = Path(arrow_file).parts
                if depth >= len(parts):
                    continue
                else:
                    arrow_file_part = '/'.join(parts[:-depth])
                    if arrow_file_part not in existed:
                        arrow_files_output_list.append(arrow_file_part)
                        existed.add(arrow_file_part)
            else:
                raise ValueError(f'Depth {depth} has exceeded the limit of arrow file {arrow_file}.')
        arrow_files_repr = '\n'.join(arrow_files_output_list)
        print(arrow_files_repr)
        return None

    return_space = '\n' + ' ' * 21

    if len(arrow_files) <= 4:
        arrow_files_repr = return_space.join([arrow_file for arrow_file in arrow_files])
    else:
        arrow_files_repr = return_space.join([_ for _ in arrow_files[:2]] + ['...']
                                             + [_ for _ in arrow_files[-2:]])
    arrow_files_length = len(arrow_files)

    # Format data_type
    data_type = data['data_type']
    if isinstance(data_type, str):
        data_type = [data_type]
    data_type_common = []
    src_files = []
    found_src_files = False
    for data_type_item in data_type:
        if not found_src_files and data_type_item.strip() != 'src_files=':
            data_type_common.append(data_type_item.strip())
            continue
        found_src_files = True
        if data_type_item.endswith('.json'):
            src_files.append(data_type_item.strip())
        else:
            data_type_common.append(data_type_item.strip())
    data_type_part2_with_ids = []
    max_id_len = len(str(len(src_files)))
    for sid, data_type_item in enumerate(src_files, start=1):
        data_type_part2_with_ids.append(f'{str(sid).rjust(max_id_len)}. {data_type_item}')
    data_type = data_type_common + data_type_part2_with_ids
    data_repr = return_space.join(data_type)

    # Format cum_length examples
    cum_length = data['cum_length']
    if len(cum_length) <= 8:
        cum_length_repr = ', '.join([str(i) for i in cum_length])
    else:
        cum_length_repr = ', '.join([str(i) for i in cum_length[:4]] + ['...'] + [str(i) for i in cum_length[-4:]])
    cum_length_length = len(cum_length)

    if detect_index_type(data) == 'base':
        # Format group_length examples
        group_length = data['group_length']
        if len(group_length) <= 8:
            group_length_repr = ', '.join([str(i) for i in group_length])
        else:
            group_length_repr = ', '.join([str(i) for i in group_length[:4]] + ['...'] + [str(i) for i in group_length[-4:]])
        group_length_length = len(group_length)

        # Format indices examples
        indices = data['indices']
        if len(indices) == 0 and data['indices_file'] != '':
            indices_file = Path(src).parent / data['indices_file']
            if Path(indices_file).exists():
                print(f"Loading indices from {indices_file} ...")
                indices = np.load(indices_file)['x']
                print(f"Loaded.")
            else:
                raise ValueError(f'This Index file contains an extra file {indices_file} which is missed.')
        if len(indices) <= 8:
            indices_repr = ', '.join([str(i) for i in indices])
        else:
            indices_repr = ', '.join([str(i) for i in indices[:4]] + ['...'] + [str(i) for i in indices[-4:]])

        # Calculate indices total length
        indices_length = len(indices)

        print_str = f"""File: {Path(src).absolute()}
        
ArrowIndexV2(
          \033[4mdata_type:\033[0m {data_repr}"""

        # Process optional data
        if opt_data['config_file'] is not None:
            print_str += f"""
        \033[4mconfig_file:\033[0m {opt_data['config_file']}"""

        # Add common data
        print_str += f"""
       \033[4mindices_file:\033[0m {data['indices_file']}
        \033[4marrow_files: Count = {arrow_files_length:,}\033[0m
                     Examples: {arrow_files_repr}
         \033[4mcum_length: Count = {cum_length_length:,}\033[0m
                     Examples: {cum_length_repr}
       \033[4mgroup_length: Count = {group_length_length:,}\033[0m
                     Examples: {group_length_repr}
            \033[4mindices: Count = {indices_length:,}\033[0m
                     Examples: {indices_repr}"""

    else:
        group_length = data['group_length']

        indices_file = Path(src).parent / data['indices_file']
        assert Path(indices_file).exists(), f'indices_file {indices_file} not found'
        print(f"Loading indices from {indices_file} ...")
        indices_data = np.load(indices_file)
        print(f"Loaded.")
        indices_length = sum([len(indices) for key, indices in indices_data.items()])
        keys = [k for k in group_length.keys() if len(indices_data[k]) > 0]

        resolutions = ResolutionGroup.from_list_of_hxw(keys)
        resolutions.attr = [f'{len(indices):>,d}' for k, indices in indices_data.items()]
        resolutions.prefix_space = 25

        print_str = f"""File: {Path(src).absolute()}

MultiResolutionBucketIndexV2(
          \033[4mdata_type:\033[0m {data_repr}"""

        # Process optional data
        if opt_data['config_file'] is not None:
            print_str += f"""
        \033[4mconfig_file:\033[0m {opt_data['config_file']}"""

        # Process config files of base index files
        config_files = []
        for src_file in src_files:
            src_file = Path(src_file)
            if src_file.exists():
                with src_file.open() as f:
                    base_data = json.load(f)
                if 'config_file' in base_data:
                    config_files.append(base_data['config_file'])
                else:
                    config_files.append('Unknown')
            else:
                config_files.append('Missing the src file')
        if config_files:
            config_file_str = return_space.join([f'{str(sid).rjust(max_id_len)}. {config_file}'
                                                 for sid, config_file in enumerate(config_files, start=1)])
            print_str += f"""
  \033[4mbase config files:\033[0m {config_file_str}"""

        # Add common data
        print_str += f"""
       \033[4mindices_file:\033[0m {data['indices_file']}
        \033[4marrow_files: Count = {arrow_files_length:,}\033[0m
                     Examples: {arrow_files_repr}
         \033[4mcum_length: Count = {cum_length_length:,}\033[0m
                     Examples: {cum_length_repr}
            \033[4mindices: Count = {indices_length:,}\033[0m
            \033[4mbuckets: Count = {len(keys)}\033[0m
                     {resolutions}"""

    print(print_str + '\n)\n')
