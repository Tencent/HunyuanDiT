# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import os.path as osp
import platform
import shutil
import tempfile
import types
import warnings
from packaging.version import parse

from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

from addict import Dict

import argparse
import deepspeed

from constants import *
from modules.models import HUNYUAN_DIT_CONFIG

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
DEPRECATION_KEY = '_deprecation_'
RESERVED_KEYS = []

def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


def _configdict2string(cfg_dict, dict_type=None):
    if isinstance(cfg_dict, dict):
        dict_type = dict_type or type(cfg_dict)
        return dict_type(
            {k: _configdict2string(v, dict_type)
             for k, v in dict.items(cfg_dict)})
    elif isinstance(cfg_dict, (tuple, list)):
        return type(cfg_dict)(_configdict2string(v, dict_type) for v in cfg_dict)
    else:
        return cfg_dict


class ConfigDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.

    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.
    """

    def __init__(__self, *args, **kwargs):
        object.__setattr__(__self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(__self, '__key', kwargs.pop('__key', None))
        object.__setattr__(__self, '__frozen', False)
        for arg in args:
            if not arg:
                continue
            if isinstance(arg, ConfigDict):
                for key, val in dict.items(arg):
                    __self[key] = __self._hook(val)
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    __self[key] = __self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                __self[arg[0]] = __self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    __self[key] = __self._hook(val)

        for key, val in dict.items(kwargs):
            __self[key] = __self._hook(val)

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value

    @classmethod
    def _hook(cls, item):
        # avoid to convert user defined dict to ConfigDict.
        if type(item) in (dict, OrderedDict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item

    def __setattr__(self, name, value):
        value = self._hook(value)
        return super().__setattr__(name, value)

    def __setitem__(self, name, value):
        value = self._hook(value)
        return super().__setitem__(name, value)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, value in super().items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return other

    def __copy__(self):
        other = self.__class__()
        for key, value in super().items():
            other[key] = value
        return other

    copy = __copy__

    def __iter__(self):
        # Implement `__iter__` to overwrite the unpacking operator `**cfg_dict`
        # to get the built object
        return iter(self.keys())

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get the value of the key.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return super().get(key, default)

    def pop(self, key, default=None):
        """Pop the value of the key.

        Args:
            key (str): The key.
            default (any, optional): The default value. Defaults to None.

        Returns:
            Any: The value of the key.
        """
        return super().pop(key, default)

    def update(self, *args, **kwargs) -> None:
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError('update only accept one positional argument')
            for key, value in dict.items(args[0]):
                other[key] = value

        for key, value in dict(kwargs).items():
            other[key] = value
        for k, v in other.items():
            if ((k not in self) or (not isinstance(self[k], dict))
                    or (not isinstance(v, dict))):
                self[k] = self._hook(v)
            else:
                self[k].update(v)

    def values(self):
        """Yield the values of the dictionary.
        """
        values = []
        for value in super().values():
            values.append(value)
        return values

    def items(self):
        """Yield the keys and values of the dictionary.
        """
        items = []
        for key, value in super().items():
            items.append((key, value))
        return items

    def merge(self, other: dict):
        """Merge another dictionary into current dictionary.

        Args:
            other (dict): Another dictionary.
        """
        default = object()

        def _merge_a_into_b(a, b):
            if isinstance(a, dict):
                if not isinstance(b, dict):
                    a.pop(DELETE_KEY, None)
                    return a
                if a.pop(DELETE_KEY, False):
                    b.clear()
                all_keys = list(b.keys()) + list(a.keys())
                return {
                    key:
                    _merge_a_into_b(a.get(key, default), b.get(key, default))
                    for key in all_keys if key != DELETE_KEY
                }
            else:
                return a if a is not default else b

        merged = _merge_a_into_b(copy.deepcopy(other), copy.deepcopy(self))
        self.clear()
        for key, value in merged.items():
            self[key] = value

    def __reduce_ex__(self, proto):
        # Override __reduce_ex__ to avoid `self.items` will be
        # called by CPython interpreter during pickling. See more details in
        # https://github.com/python/cpython/blob/8d61a71f9c81619e34d4a30b625922ebc83c561b/Objects/typeobject.c#L6196  # noqa: E501
        if digit_version(platform.python_version()) < digit_version('3.8'):
            return (self.__class__, ({k: v
                                      for k, v in super().items()}, ), None,
                    None, None)
        else:
            return (self.__class__, ({k: v
                                      for k, v in super().items()}, ), None,
                    None, None, None)

    def __eq__(self, other):
        if isinstance(other, ConfigDict):
            return other.to_dict() == self.to_dict()
        elif isinstance(other, dict):
            return {k: v for k, v in self.items()} == other
        else:
            return False

    def to_dict(self):
        """Convert the ConfigDict to a normal dictionary recursively."""
        return _configdict2string(self, dict_type=dict)


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == self.key):
            return None
        else:
            return node


class ConfigParsingError(RuntimeError):
    """Raise error when failed to parse pure Python style config files."""


class Config:

    def __init__(self,
                 cfg_dict: dict = None,
                 cfg_text: Optional[str] = None,
                 filename: Optional[Union[str, Path]] = None):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        if not isinstance(cfg_dict, ConfigDict):
            cfg_dict = ConfigDict(cfg_dict)

        super().__setattr__('_cfg_dict', cfg_dict)
        super().__setattr__('_filename', filename)

        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)

    @staticmethod
    def _validate_py_syntax(filename: str):
        """Validate syntax of python config.

        Args:
            filename (str): Filename of python config file.
        """
        with open(filename, encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')


    @staticmethod
    def _get_base_files(filename: str) -> list:
        """Get the base config file.

        Args:
            filename (str): The config file.

        Raises:
            TypeError: Name of config file.

        Returns:
            list: A list of base config.
        """
        file_format = osp.splitext(filename)[1]
        if file_format == '.py':
            Config._validate_py_syntax(filename)
            with open(filename, encoding='utf-8') as f:
                parsed_codes = ast.parse(f.read()).body

                def is_base_line(c):
                    return (isinstance(c, ast.Assign)
                            and isinstance(c.targets[0], ast.Name)
                            and c.targets[0].id == BASE_KEY)

                base_code = next((c for c in parsed_codes
                                  if is_base_line(c)),
                                  None)
                if base_code is not None:
                    base_code = ast.Expression(  # type: ignore
                        body=base_code.value)  # type: ignore
                    base_files = eval(compile(base_code, '', mode='eval'))
                else:
                    base_files = []
        else:
            raise ConfigParsingError(
                'The config type should be py, '
                'but got {file_format}')
        base_files = base_files if isinstance(base_files,
                                              list) else [base_files]
        return base_files
    
    @staticmethod
    def _get_cfg_path(cfg_path: str,
                      filename: str) -> Tuple[str, Optional[str]]:
        """Get the config path from the current or external package.

        Args:
            cfg_path (str): Relative path of config.
            filename (str): The config file being parsed.

        Returns:
            Tuple[str, str or None]: Path and scope of config. If the config
            is not an external config, the scope will be `None`.
        """
        # Get local config path.
        cfg_dir = osp.dirname(filename)
        cfg_path = osp.join(cfg_dir, cfg_path)
        return cfg_path, None

    @staticmethod
    def _dict_to_config_dict(cfg: dict,
                             scope: Optional[str] = None,
                             has_scope=True):
        """Recursively converts ``dict`` to :obj:`ConfigDict`.

        Args:
            cfg (dict): Config dict.
            scope (str, optional): Scope of instance.
            has_scope (bool): Whether to add `_scope_` key to config dict.

        Returns:
            ConfigDict: Converted dict.
        """
        # Only the outer dict with key `type` should have the key `_scope_`.
        if isinstance(cfg, dict):
            if has_scope and 'type' in cfg:
                has_scope = False
                if scope is not None and cfg.get('_scope_', None) is None:
                    cfg._scope_ = scope  # type: ignore
            cfg = ConfigDict(cfg)
            dict.__setattr__(cfg, 'scope', scope)
            for key, value in cfg.items():
                cfg[key] = Config._dict_to_config_dict(
                    value, scope=scope, has_scope=has_scope)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._dict_to_config_dict(_cfg, scope, has_scope=has_scope)
                for _cfg in cfg
            ]
        return cfg

    
    @staticmethod
    def _merge_a_into_b(a: dict,
                        b: dict,
                        allow_list_keys: bool = False) -> dict:
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Defaults to False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types: Union[Tuple, type] = (
                        dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    @staticmethod
    def _file2dict(filename: str) -> Tuple[dict, str, dict]:
        """Transform file to variables dictionary.
        Args:
            filename (str): Name of config file.
        Returns:
            Tuple[dict, str]: Variables dictionary and text of Config.
        """

        filename = osp.abspath(osp.expanduser(filename))
        if not os.path.exists(filename):
            raise FileNotFoundError(f'{filename} is not exist!')
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py']:
            raise OSError('Only py type are supported now!')
        try:
            with tempfile.TemporaryDirectory() as temp_config_dir:
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix=fileExtname, delete=False)
                if platform.system() == 'Windows':
                    temp_config_file.close()
                
                shutil.copyfile(filename, temp_config_file.name)

                # Handle base files
                base_cfg_dict = ConfigDict()
                cfg_text_list = list()
                
                for base_cfg_path in Config._get_base_files(
                    temp_config_file.name):
                    base_cfg_path, scope = Config._get_cfg_path(
                        base_cfg_path, filename)
                    # Generate base config
                    _cfg_dict, _cfg_text = Config._file2dict(
                        filename=base_cfg_path)
                    cfg_text_list.append(_cfg_text)
                    # Check duplicate
                    duplicate_keys = base_cfg_dict.keys() & _cfg_dict.keys()
                    if len(duplicate_keys) > 0:
                        raise KeyError(
                            'Duplicate key is not allowed among bases. '
                            f'Duplicate keys: {duplicate_keys}')

                    # _dict_to_config_dict will do the following things:
                    # 1. Recursively converts ``dict`` to :obj:`ConfigDict`.
                    # 2. Set `_scope_` for the outer dict variable for the base
                    # config.
                    # 3. Set `scope` attribute for each base variable.
                    # Different from `_scope_`, `scope` is not a key of base
                    # dict, `scope` attribute will be parsed to key `_scope_`
                    # by function `_parse_scope` only if the base variable is
                    # accessed by the current config.
                    _cfg_dict = Config._dict_to_config_dict(_cfg_dict, scope)
                    base_cfg_dict.update(_cfg_dict)

                if filename.endswith('.py'):
                    with open(temp_config_file.name, encoding='utf-8') as f:
                        parsed_codes = ast.parse(f.read())
                        parsed_codes = RemoveAssignFromAST(BASE_KEY).visit(parsed_codes)
                    codeobj = compile(parsed_codes, filename, mode='exec')
                    # Support load global variable in nested function of the config.
                    global_locals_var = {BASE_KEY: base_cfg_dict}
                    ori_keys = set(global_locals_var.keys())
                    eval(codeobj, global_locals_var, global_locals_var)
                    cfg_dict = {
                        key: value
                        for key, value in global_locals_var.items()
                        if (key not in ori_keys and not key.startswith('__'))
                    }
                # close temp file
                for key, value in list(cfg_dict.items()):
                    if isinstance(value,
                                  (types.FunctionType, types.ModuleType)):
                        cfg_dict.pop(key)
                temp_config_file.close()

        except Exception as e:
            if osp.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            raise e

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = f'The config file {filename} will be deprecated ' \
                'in the future.'
            if 'expected' in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' \
                    'instead.'
            if 'reference' in deprecation_info:
                warning_msg += ' More information can be found at ' \
                    f'{deprecation_info["reference"]}'
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + '\n'
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        cfg_dict.pop(BASE_KEY, None)

        cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
        cfg_dict = {
            k: v
            for k, v in cfg_dict.items() if not k.startswith('__')
        }

        # merge cfg_text
        cfg_text_list.append(cfg_text)
        cfg_text = '\n'.join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def fromfile(filename: Union[str, Path]) -> 'Config':
        """Build a Config instance from config file.

        Args:
            filename (str or Path): Name of config file.
            format_python_code (bool): Whether to format Python code by yapf.
                Defaults to True.

        Returns:
            Config: Config instance built from config file.
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        cfg_dict, cfg_text = Config._file2dict(filename)

        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __repr__(self):
        return f'Config (path: {self._filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(
            self
    ) -> Tuple[dict, Optional[str], Optional[str], dict, bool, set]:
        state = (self._cfg_dict, self._filename, self._text)
        return state

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)
        super(Config, other).__setattr__('_cfg_dict', self._cfg_dict.copy())

        return other

    copy = __copy__

    def __setstate__(self, state: Tuple[dict, Optional[str], Optional[str],
                                        dict, bool, set]):
        super().__setattr__('_cfg_dict', state[0])
        super().__setattr__('_filename', state[1])
        super().__setattr__('_text', state[2])

    def to_dict(self):
        """
        Convert all data in the config to a builtin ``dict``.
        """
        cfg_dict = self._cfg_dict.to_dict()
        return cfg_dict

config_rules = {
    'model': {
        'choices': list(HUNYUAN_DIT_CONFIG.keys())
    },
    'norm': {
        'choices': ["rms", "layer"]
    },
    'training_parts': {
        'choices': ['all', 'lora']
    },
    'control_type': {
        'choices': ['canny', 'depth', 'pose']
    },
    'predict_type': {
        'choices': list(PREDICT_TYPE)
    },
    'noise_schedule': {
        'choices': list(NOISE_SCHEDULES)
    },
    'load_key': {
        'choices': ["ema", "module", "distill", 'merge']
    },
    'infer_mode': {
        'choices': ["fa", "torch", "trt"]
    },
    'sampler': {
        'choices': SAMPLER_FACTORY
    },
    'lang': {
        'choices': ["zh", "en"]
    },
    'rope_img': {
        'choices': ['extend', 'base512', 'base1024']
    },
    'ema_dtype': {
        'choices': ['fp16', 'fp32', 'none']
    },
    'remote_device': {
        'choices': ['none', 'cpu', 'nvme']
    }
}

def get_args(default_args=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args(default_args)

    config = Config.fromfile(args.config)
    # Check config values
    for key, value in config.items():
        if key in config_rules:
            rule = config_rules[key]
            if 'choices' in rule and value not in rule['choices']:
                raise ValueError(f"Invalid value '{value}' for '{key}'. Choices are: {rule['choices']}")
    # Merge parsing argement
    for key, value in args.__dict__.items():
        config[key] = value

    return config

if __name__ == '__main__':
    print(get_args())
