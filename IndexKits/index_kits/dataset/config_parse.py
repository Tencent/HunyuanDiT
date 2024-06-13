import yaml
from glob import glob
import pathlib
from pathlib import Path
import pickle
import json
from typing import List
from collections import defaultdict

def source_names_to_names(source_names_):
    """
    Get all arrow file names from a list of source names.

    Examples:
    `test.arrow@@x2` means repeat the arrow file `test.arrow` 2 times.
    """
    names = []
    exclude_count = 0
    for data_name in source_names_:
        if isinstance(data_name, dict):
            data_name, value = list(data_name.items())[0]
            exclude = value.get('exclude', [])
            repeat_times = value.get('repeat', 1)
        else:
            exclude = []
            repeat_times = 1

        cur_count = 0
        for name in glob(data_name):
            for e in exclude:
                if e in name:
                    print(f"Excluding {name}")
                    exclude_count += 1
                    break
            else:
                names.append((name, repeat_times))
                cur_count += 1
        print(f"Found {cur_count} arrow files in {data_name}.")
    names = list(sorted(names, key=lambda x: x[0]))
    return names, exclude_count


def get_or_create_arrow_name_list(source_names=None):

    if not isinstance(source_names, list):
        raise ValueError(f"`source_names` should be of type list, got {type(source_names)}")
    if len(source_names) == 0:
        raise ValueError(f"`source_names` is an empty list.")
    names, exclude_count = source_names_to_names(source_names)

    print(f"Found {len(names):,} arrow files.")
    print(f"Excluded {exclude_count} arrow files.")
    return names


def to_numeric(x, default_val=0.0):
    if isinstance(x, (float, int)):
        pass
    elif isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            x = default_val
    else:
        x = default_val
    return x


TYPE_MAPPER = {
    "int": int,
    "float": float,
    "str": str,
    "dict": dict,
    "list": list,
}


class Operator(object):
    def __init__(self, config):
        self.config = config
        self.arrow_file_keyword = None

    def applicable(self, arrow_file):
        # If no keyword is provided, then it is applicable to all arrow files
        if self.arrow_file_keyword is None:
            return True
        # If keyword is provided, then it is applicable to the arrow file if the keyword is in the arrow file name
        for keyword in self.arrow_file_keyword:
            if keyword in arrow_file:
                return True
        return False

    @staticmethod
    def parse_path(paths):
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            raise ValueError(f"Invalid type: {type(paths)}")
        paths = [Path(p) for p in paths]
        return paths

    @staticmethod
    def load_md5s(paths: List[pathlib.Path], merge=True):
        md5s_list = []
        for path in paths:
            if not path.exists():
                raise ValueError(f"Path not found: {path}")
            if path.suffix == '.pkl':
                with path.open('rb') as f:
                    md5s = pickle.load(f)
                assert isinstance(md5s, (set, dict)), f"Invalid type: {type(md5s)}"
            elif path.suffix == '.json':
                with path.open() as f:
                    md5s = json.load(f)
                assert isinstance(md5s, dict), f"Invalid type: {type(md5s)}"
            elif path.suffix == '.txt':
                md5s = set(path.read_text().splitlines())
            else:
                raise ValueError(f"Invalid file type: {path}")
            md5s_list.append(md5s)
            print(f"    {path}: {len(md5s):,} md5s")

        if merge:
            md5s = md5s_list[0]
            for m in md5s_list[1:]:
                md5s.update(m)
        else:
            md5s = md5s_list
        return md5s


class ColumnFilter(Operator):
    numeric_actions = {"eq", "ne", "gt", "lt", "ge", "le"}
    str_actions = {"eq", "ne", "len_eq", "len_ne", "len_gt", "len_lt", "len_ge", "len_le", "contains", "not_contains",
                   "in", "not_in", "lower_last_in"}
    int_target_str_actions = {"len_eq", "len_gt", "len_lt", "len_ge", "len_le"}
    action_mapper_left = {
        "eq": "==", "ne": "!=", "len_eq": ".len()==",
        "gt": ">", "len_gt": ".len()>",
        "lt": "<", "len_lt": ".len()<",
        "ge": ">=", "len_ge": ".len()>=",
        "le": "<=", "len_le": ".len()<=",
        "contains": ".contains(",
        "not_contains": ".not_contains(",
        "in": ".isin(", "not_in": ".notin(",
        "lower_last_in": "[-1].lower().isin(",
    }
    action_mapper_right = {
        "eq": "", "ne": "", "len_eq": "",
        "gt": "", "len_gt": "",
        "lt": "", "len_lt": "",
        "ge": "", "len_ge": "",
        "le": "", "len_le": "",
        "contains": ")",
        "not_contains": ")",
        "in": ")", "not_in": ")",
        "lower_last_in": ")",
    }

    def __init__(self, config):
        super().__init__(config)
        self.name = self.column_name = config['name']
        self.dtype = config['type']
        self.action = config['action']
        self.target = config['target']
        self.default = config['default']
        self.arrow_file_cond = config.get('arrow_file', None)
        self.arrow_file_keyword = config.get('arrow_file_keyword', None)
        if self.arrow_file_keyword is not None:
            if isinstance(self.arrow_file_keyword, str):
                self.arrow_file_keyword = [self.arrow_file_keyword]

    def check_exists(self, arrow_file, table):
        status = True
        if self.column_name not in table.column_names:
            print(f"Warning: Column `{self.column_name}` not found in {arrow_file}.")
            status = status and False
        return status

    @property
    def data_type(self):
        if self.arrow_file_keyword is None or len(self.arrow_file_keyword) == 0:
            arrow_file_keyword_repr = ""
        else:
            arrow_file_keyword_repr = f', arrow_file_keys={self.arrow_file_keyword}'
        fmt_str = f"{self.column_name}" \
                  f"{self.action_mapper_left[self.action]}{self.target}{self.action_mapper_right[self.action]}" \
                  f" (default={self.default}{arrow_file_keyword_repr})"
        return fmt_str


class NumericFilter(ColumnFilter):
    def run_on_column(self, column):
        if self.action == 'eq':
            return column == self.target
        elif self.action == 'ne':
            return column != self.target
        elif self.action == 'gt':
            return column > self.target
        elif self.action == 'lt':
            return column < self.target
        elif self.action == 'ge':
            return column >= self.target
        elif self.action == 'le':
            return column <= self.target
        else:
            raise ValueError(f"Invalid action: {self.action}")


class IntFilter(NumericFilter):
    def __call__(self, arrow_file, table):
        success = self.check_exists(arrow_file, table)
        if not success:
            return None
        column = table[self.column_name].to_pandas().apply(lambda x: to_numeric(x, self.default))
        return self.run_on_column(column)


class FloatFilter(NumericFilter):
    def __call__(self, arrow_file, table):
        success = self.check_exists(arrow_file, table)
        if not success:
            return None
        column = table[self.column_name].to_pandas().apply(lambda x: to_numeric(x, self.default))
        return self.run_on_column(column)


class StrFilter(ColumnFilter):
    def __call__(self, arrow_file, table):
        success = self.check_exists(arrow_file, table)
        if not success:
            return None
        column = table[self.column_name].to_pandas()
        if self.action == 'eq':
            return column.apply(lambda x: x == self.target)
        elif self.action == 'ne':
            return column.apply(lambda x: x != self.target)
        elif self.action == 'len_eq':
            return column.str.len() == self.target
        elif self.action == 'len_ne':
            return column.str.len() != self.target
        elif self.action == 'len_gt':
            return column.str.len() > self.target
        elif self.action == 'len_lt':
            return column.str.len() < self.target
        elif self.action == 'len_ge':
            return column.str.len() >= self.target
        elif self.action == 'len_le':
            return column.str.len() <= self.target
        elif self.action == 'contains':
            return column.str.contains(self.target)
        elif self.action == 'not_contains':
            return ~column.str.contains(self.target)
        elif self.action == 'in':
            return column.apply(lambda x: x in self.target)
        elif self.action == 'not_in':
            return column.apply(lambda x: x not in self.target)
        elif self.action == 'lower_last_in':
            return column.apply(lambda x: x[-1].lower() in self.target)
        else:
            raise ValueError(f"Invalid action: {self.action}")


def check_type(parent, config):
    if 'type' not in config:
        return False, f"Missing required argument: {parent}.type"
    if not isinstance(config['type'], str):
        return False, f"Argument {parent}.type should be of type str"
    if config['type'] not in TYPE_MAPPER:
        return False, f"Invalid type: {parent}.type: {config['type']}"
    return True, ""


def check_keys(parent, config, required_args, types):
    for arg, ts in zip(required_args, types):
        # Check key exists
        if arg not in config:
            return False, f"Missing required argument: {parent}.{arg}"
        # Check values' type
        if ts is not None:
            if not isinstance(ts, list):
                ts = [ts]
            for t in ts:
                if not isinstance(config[arg], t):
                    return False, f"Argument {parent}.{arg} should be of type {t}, got {type(config[arg])}"
    return True, ""


def get_column_filter(parent, config):
    success, msg = check_type(parent, config)
    if not success:
        return False, msg, None

    if config['type'] == 'str' and config['action'] in ColumnFilter.int_target_str_actions:
        dtype = TYPE_MAPPER['int']
    else:
        dtype = TYPE_MAPPER[config['type']]

    # Check other keys
    required_args = ['name', 'action', 'target', 'default']
    types = [str, str, dtype, dtype]
    success, msg = check_keys(parent, config, required_args, types)
    if not success:
        return False, msg, None

    # Check action values
    if config['type'] == 'str':
        valid_actions = ColumnFilter.str_actions
    else:
        valid_actions = ColumnFilter.numeric_actions
    if config['action'] not in valid_actions:
        return False, f"Invalid action: {parent}.action: {config['action']}", None

    if config['type'] == 'int':
        return True, "", IntFilter(config)
    elif config['type'] == 'float':
        return True, "", FloatFilter(config)
    elif config['type'] == 'str':
        return True, "", StrFilter(config)
    else:
        raise ValueError(f"Invalid type: {config['type']}")


class MD5Filter(Operator):
    valid_actions = {
        "list": {"in", "not_in"},
        "dict": {"eq", "ne", "gt", "lt", "ge", "le"},
    }
    action_mapper_left = {
        "eq": "==", "ne": "!=",
        "gt": ">", "lt": "<",
        "ge": ">=", "le": "<=",
        "in": ".isin(", "not_in": ".isin(",
    }
    action_mapper_right = {
        "eq": "", "ne": "",
        "gt": "", "lt": "",
        "ge": "", "le": "",
        "in": ")", "not_in": ")",
    }

    def __init__(self, config):
        super().__init__(config)
        self.name = config['name']
        self.paths = self.parse_path(config['path'])
        self.dtype = config['type']
        self.action = config['action']
        self.target = config.get('target', None)                            # only for type=='dict'
        self.is_valid = config.get('is_valid', None)                        # only for type=='dict'
        self.arrow_file_keyword = config.get('arrow_file_keyword', None)
        if self.arrow_file_keyword is not None:
            if isinstance(self.arrow_file_keyword, str):
                self.arrow_file_keyword = [self.arrow_file_keyword]

        self.md5_data = self.load_md5s(self.paths)

    @property
    def data_type(self):
        if self.arrow_file_keyword is None or len(self.arrow_file_keyword) == 0:
            arrow_file_keyword_repr = ""
        else:
            arrow_file_keyword_repr = f'(arrow_file_keys={self.arrow_file_keyword})'
        return_space = '\n' + ' ' * 22
        if self.dtype == 'list':
            fmt_str = ("Good Cases" if self.is_valid else " Bad Cases") \
                      + f" (md5): {return_space.join([str(p) for p in self.paths])} " \
                        f"{arrow_file_keyword_repr} " \
                        f"| {self.name}"
        elif self.dtype == 'dict':
            fmt_str = ("Good Cases" if self.is_valid else " Bad Cases") \
                      + f" (md5): {return_space.join([str(p) for p in self.paths])}\n" \
                        f"                      --> value" \
                        f"{self.action_mapper_left[self.action]}{self.target}{self.action_mapper_right[self.action]} " \
                        f"{arrow_file_keyword_repr} " \
                        f"| {self.name}"
        else:
            raise ValueError(f"Invalid type: {self.dtype}")
        return fmt_str


class ListFilter(MD5Filter):
    def __call__(self, md5s):
        if self.action == "in":
            find_valid = self.is_valid
        elif self.action == "not_in":
            find_valid = not self.is_valid
        else:
            raise ValueError(f"Invalid action: {self.action}")

        if find_valid:
            return md5s.apply(lambda x: x in self.md5_data)
        else:
            return md5s.apply(lambda x: x not in self.md5_data)


class DictFilter(MD5Filter):
    def __call__(self, md5s):
        if self.is_valid:
            return md5s.apply(lambda x: x in self.md5_data and self.cond(self.md5_data[x]))
        else:
            return md5s.apply(lambda x: (x not in self.md5_data) or not self.cond(self.md5_data[x]))

    def cond(self, value):
        if self.action == "eq":
            return value == self.target
        elif self.action == "ne":
            return value != self.target
        elif self.action == "gt":
            return value > self.target
        elif self.action == "lt":
            return value < self.target
        elif self.action == "ge":
            return value >= self.target
        elif self.action == "le":
            return value <= self.target
        else:
            raise ValueError(f"Invalid action: {self.action}")


def get_md5_filter(parent, config):
    success, msg = check_type(parent, config)
    if not success:
        return False, msg, None

    required_args = ['name', 'path', 'action']
    types = [str, (str, list), str]
    if config['type'] == 'dict':
        required_args.extend(['target', 'is_valid'])
        types.extend([None, bool])
    success, msg = check_keys(parent, config, required_args, types)
    if not success:
        return False, msg, None

    if config['action'] not in MD5Filter.valid_actions[config['type']]:
        return False, f"Invalid action: {parent}.action: {config['action']}", None

    if config['type'] == 'list':
        return True, "", ListFilter(config)
    elif config['type'] == 'dict':
        return True, "", DictFilter(config)
    else:
        raise ValueError(f"Invalid type: {config['type']}")


class FilterCompose(object):
    def __init__(self, column_filter_list, md5_filter_list):
        self.column_filter_list = column_filter_list
        self.md5_filter_list = md5_filter_list

    def __call__(self, arrow_file, table):
        stats = {}
        length = len(table)
        assert length > 0, "Empty table"

        mask = None
        for filter_ in self.column_filter_list:
            if isinstance(filter_, tuple):
                op, filter_list = filter_
                if op == 'logical_or':
                    sub_mask = None
                    for sub_filter in filter_list:
                        if not sub_filter.applicable(arrow_file):
                            continue
                        sub_current_mask = sub_filter(arrow_file, table)
                        if sub_current_mask is not None:
                            if sub_mask is None:
                                sub_mask = sub_current_mask
                            else:
                                sub_mask = sub_mask | sub_current_mask
                    if sub_mask is not None:
                        name = '|'.join([f.name for f in filter_list])
                        stats.update({
                            name: length - sum(sub_mask)
                        })
                        if mask is None:
                            mask = sub_mask
                        else:
                            mask = mask & sub_mask
                else:
                    raise ValueError(f"Invalid operation: {op}")
            else:
                if not filter_.applicable(arrow_file):
                    continue
                current_mask = filter_(arrow_file, table)
                if current_mask is not None:
                    stats.update({
                        filter_.name: length - sum(current_mask)
                    })
                    if mask is None:
                        mask = current_mask
                    else:
                        mask = mask & current_mask

        md5s = None
        for filter_ in self.md5_filter_list:
            if not filter_.applicable(arrow_file):
                continue
            if 'md5' not in table.column_names:
                print(f"Warning: Column 'md5' not found in {arrow_file}.")

            md5s = table['md5'].to_pandas()
            current_mask = filter_(md5s)
            if current_mask is not None:
                stats.update({
                    filter_.name: length - sum(current_mask)
                })
                if mask is None:
                    mask = current_mask
                else:
                    mask = mask & current_mask

        return mask, stats, md5s

    @property
    def data_type(self):
        data_type_list = []
        for filter_ in self.column_filter_list + self.md5_filter_list:
            if isinstance(filter_, tuple):
                data_type_list.append(' || '.join([f.data_type for f in filter_[1]]))
            else:
                data_type_list.append(filter_.data_type)
        return data_type_list


class MD5Repeater(Operator):
    def __init__(self, config):
        super().__init__(config)
        self.name = config['name']
        self.paths = self.parse_path(config['path'])
        self.dtype = config['type']
        self.plus = config.get('plus', 0)
        self.repeat = config.get('repeat', None)
        self.arrow_file_keyword = config.get('arrow_file_keyword', None)
        if self.arrow_file_keyword is not None:
            if isinstance(self.arrow_file_keyword, str):
                self.arrow_file_keyword = [self.arrow_file_keyword]

        self.md5_data = self.load_md5s(self.paths)
        if self.repeat is None:
            # Check if md5_data.values() are integers
            for v in self.md5_data.values():
                if not isinstance(v, int):
                    raise ValueError(f"Values from {self.paths} are not integers. For example: {v}")
                # We only check the first value for performance
                # We assume all values are the same type
                break

    def __call__(self, arrow_file, index, md5):
        if md5 in self.md5_data:
            if self.repeat is None:
                return self.md5_data[md5] + self.plus
            else:
                return self.repeat
        else:
            return 1

    @property
    def data_type(self):
        path_repr = ', '.join([str(p) for p in self.paths])
        if self.dtype == 'list':
            fmt_str = f"[MD5] Repeat {self.repeat} times: {path_repr}"
        elif self.dtype == 'dict':
            fmt_str = f"[MD5] Repeat multiple times: {path_repr}"
        else:
            raise ValueError(f"Invalid type: {self.dtype}")
        return fmt_str


def get_md5_repeater(parent, config):
    success, msg = check_type(parent, config)
    if not success:
        return False, msg, None

    required_args = ['name', 'path']
    types = [str, str]
    if config['type'] == 'list':
        required_args.extend(['repeat'])
        types.extend([int])
    success, msg = check_keys(parent, config, required_args, types)
    if not success:
        return False, msg, None

    return True, "", MD5Repeater(config)


class KeywordRepeater(Operator):
    def __init__(self, config):
        super().__init__(config)
        self.keywords = config['keyword']
        self.repeat = config['repeat']
        self.name = f"Repeat {self.repeat} times"

    def __call__(self, arrow_file, idx, md5):
        for key in self.keywords:
            if key in arrow_file:
                return self.repeat
        return 1

    @property
    def data_type(self):
        fmt_str = f"[Keyword] Repeat {self.repeat} times: {self.keywords}"
        return fmt_str

def get_keyword_repeater(parent, config):
    required_args = ["repeat", "keyword"]
    types = [int, list]
    success, msg = check_keys(parent, config, required_args, types)
    if not success:
        return False, msg, None

    return True, "", KeywordRepeater(config)


class RepeaterCompose(object):
    def __init__(self, repeater_list):
        self.repeater_list = repeater_list

    def __call__(self, arrow_file, table, indices, repeat_times=1, md5s=None):
        stats = defaultdict(int)
        length = len(table)
        assert length > 0, "Empty table"

        if md5s is None:
            if 'md5' not in table.column_names:
                print(f"Warning: Column 'md5' not found in {arrow_file}.")
                md5s = None
            else:
                md5s = table['md5'].to_pandas()

        repeated_indices = []
        for idx in indices:
            md5 = md5s[idx] if md5s is not None else None
            max_repeat = repeat_times
            max_i = -1
            for i, repeater in enumerate(self.repeater_list):
                if not repeater.applicable(arrow_file):
                    continue
                repeat = repeater(arrow_file, idx, md5)
                if repeat > max_repeat:
                    max_repeat = repeat
                    max_i = i
            if max_i >= 0:
                stats[self.repeater_list[max_i].name] += (max_repeat - 1)
            repeated_indices.extend([idx] * max_repeat)

        return repeated_indices, stats

    @property
    def data_type(self):
        data_type_list = []
        for repeater in self.repeater_list:
            data_type_list.append(repeater.data_type)
        return data_type_list


class DatasetConfig(object):
    def __init__(self, work_dir, config_file):
        self.work_dir = work_dir
        self.config_file = str(Path(config_file).absolute())

        with open(self.config_file, 'r') as f:
            self.data = yaml.safe_load(f)

        self.names = self.parse_names()             # arrow names
        arrow_max_repeat = max([x[1] for x in self.names])
        self.filter = self.parse_filter()
        self.repeater = self.parse_repeater(enable_arrow_repeat=arrow_max_repeat > 1)

        # Extra arguments
        self.remove_md5_dup = self.data.get('remove_md5_dup', False)

    def assert_unknown_args(self):
        unknown_args = set(self.data.keys()) - {
            'source', 'filter', 'repeater', 'remove_md5_dup'
        }
        if len(unknown_args) > 0:
            raise ValueError(f"Unknown arguments in config file ({self.config_file}): {unknown_args}")

    def parse_filter(self):
        column_filter_list = []
        md5_filter_list = []
        if 'filter' in self.data:
            filters = self.data['filter']
            if 'column' in filters:
                column_filter_configs = filters['column']
                assert isinstance(column_filter_configs, list), "filter.column should be a list."
                for i, config in enumerate(column_filter_configs):
                    if config.get('logical_or', None) is not None:
                        assert isinstance(config['logical_or'], list), \
                            f"filter.column[{i}].logical_or should be a list, got {type(config['logical_or'])}"
                        sub_column_filter_list = []
                        for j, sub_config in enumerate(config['logical_or']):
                            sub_success, sub_msg, sub_filter_ = get_column_filter(f'filter.column[{i}-logical_or].[{j}]', sub_config)
                            if not sub_success:
                                raise ValueError(sub_msg)
                            sub_column_filter_list.append(sub_filter_)
                        success = True
                        msg = ''
                        filter_ = ('logical_or', sub_column_filter_list)
                    else:
                        success, msg, filter_ = get_column_filter(f'filter.column[{i}]', config)
                    if not success:
                        raise ValueError(msg)
                    column_filter_list.append(filter_)
            if 'md5' in filters:
                md5_filter_configs = filters['md5']
                assert isinstance(md5_filter_configs, list), "filter.md5 should be a list."
                for i, config in enumerate(md5_filter_configs):
                    if config.get('logical_or', None) is not None:
                        assert isinstance(config['logical_or'], list), \
                            f"filter.md5[{i}].logical_or should be a list, got {type(config['logical_or'])}"
                        sub_md5_filter_list = []
                        for j, sub_config in enumerate(config['logical_or']):
                            sub_success, sub_msg, sub_filter_ = get_md5_filter(f'filter.md5[{i}-logical_or].[{j}]', sub_config)
                            if not sub_success:
                                raise ValueError(sub_msg)
                            sub_md5_filter_list.append(sub_filter_)
                        success = True
                        msg = ''
                        filter_ = ('logical_or', sub_md5_filter_list)
                    else:
                        success, msg, filter_ = get_md5_filter(f'filter.md5[{i}]', config)
                    if not success:
                        raise ValueError(msg)
                    md5_filter_list.append(filter_)

        if column_filter_list or md5_filter_list:
            composed_filter = FilterCompose(column_filter_list, md5_filter_list)
        else:
            composed_filter = None
        return composed_filter

    def parse_repeater(self, enable_arrow_repeat=False):
        repeater_list = []
        if 'repeater' in self.data:
            repeaters = self.data['repeater']
            if 'md5' in repeaters:
                md5_repeater_configs = repeaters['md5']
                assert isinstance(md5_repeater_configs, list), "repeater.md5 should be a list."
                for i, config in enumerate(md5_repeater_configs):
                    success, msg, repeater = get_md5_repeater(f'repeater.md5[{i}]', config)
                    if not success:
                        raise ValueError(msg)
                    repeater_list.append(repeater)

            if 'arrow_file_keyword' in repeaters:
                keyword_repeater_configs = repeaters['arrow_file_keyword']
                assert isinstance(keyword_repeater_configs, list), "repeater.arrow_file_keyword should be a list."
                for i, config in enumerate(keyword_repeater_configs):
                    success, msg, repeater = get_keyword_repeater(f'repeater.arrow_file_keyword[{i}]', config)
                    if not success:
                        raise ValueError(msg)
                    repeater_list.append(repeater)

        if repeater_list or enable_arrow_repeat:
            composed_repeater = RepeaterCompose(repeater_list)
        else:
            composed_repeater = None
        return composed_repeater

    def parse_names(self):
        if 'source' in self.data:
            source = self.data['source']
            assert isinstance(source, list), "source should be a list."
        else:
            raise ValueError("In the YAML file, the ‘source’ field is filled in incorrectly, please check")

        names = get_or_create_arrow_name_list(source)
        return names

    @property
    def data_type(self):
        data_types = {
            'Filters': [] if self.filter is None else self.filter.data_type,
            'Repeaters': [] if self.repeater is None else self.repeater.data_type,
        }
        return data_types
