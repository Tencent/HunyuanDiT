# Use the IndexKits library to create datasets.

[TOC]

## Introdution

The IndexKits library offers a command-line tool `idk` for creating datasets and viewing their statistical information. You can view the instructions by using `idk -h`. Here, it refers to creating an Index V2 format dataset from a series of arrow files.

## 1. Creating a Base Index V2 Dataset

### 1.1 Create a Base Index V2 dataset using `idk`

When creating a Base Index V2 dataset, you need to specify the path to a configuration file using the `-c` parameter and a save path using the `-t` parameter.

```shell
idk base -c base_config.yaml -t base_dataset.json
```

### 1.2 Basic Configuration

Next, let’s discuss how to write the configuration file. The configuration file is in `yaml` format, and below is a basic example:

Filename: `base_config.yaml`

```yaml
source:
    - /HunYuanDiT/dataset/porcelain/arrows/00000.arrow
```

|    Field Name    | Type  |     Description     |
|:---------:|:---:|:----------:|
|  source   | Optional  |  Arrow List  |


We provide an example that includes all features and fields in [full_config.yaml](./docs/full_config.yaml).

### 1.3 Filtering Criteria

`idk` offers two types of filtering capabilities during the dataset creation process: (1) based on columns in Arrow, and (2) based on MD5 files. 

To enable filtering criteria, add the `filter` field in the configuration file.

#### 1.3.1 Column Filtering

To enable column filtering, add the `column` field under the `filter` section. 
Multiple column filtering criteria can be applied simultaneously, with the intersection of multiple conditions being taken.

For example, to select data where both the length and width are greater than or equal to 512, 
with the default being 1024 if the length and width are invalid:

```yaml
filter:
    column:
        -   name: height
            type: int
            action: ge
            target: 512
            default: 1024
        -   name: width
            type: int
            action: ge
            target: 512
            default: 1024
```

This filtering condition is equivalent to `table['height'].to_int(default=1024) >= 512 && table['width'].to_int(default=1024) >= 512`.

Each filtering criterion includes the following fields:

|      Field Name    | Type  |   Description     |          Value Range          |
|:------------------:|:---:|:---------------:|:----------------------:|
|        name        | Required  |   Column name in Arrow    |       Column in Arrow       |
|        type        | Required  |    Type of Elements in the Column     | `int`, `float` or `str` |
|       action       | Required  |     Filtering Criteria     |    See the table below for possible values      |
|       target       | Required  |    Filtering Criteria    |       Numeric or String       |
|      default       | Required  | Default value when the element is invalid  |       Numeric or String       |
| arrow_file_keyword | Optional  | Keywords in the Arrow file path |           -            |

Below are the specific meanings of “action” and the optional values under different circumstances::

|    Action     |                     Description                      |         Type          |    Action    |                 Description                  |         Type          |
|:-------------:|:----------------------------------------------------:|:---------------------:|:------------:|:--------------------------------------------:|:---------------------:|
|      eq       |                     equal, `==`                      | `int`, `float`, `str` |      ne      |               not equal, `!=`                | `int`, `float`, `str` |
|      gt       |                   great than, `>`                    |    `int`, `float`     |      lt      |                less than, `<`                |    `int`, `float`     |
|      ge       |                 great or equal, `>=`                 |    `int`, `float`     |      le      |             less or equal, `<=`              |    `int`, `float`     |
|    len_eq     |           str length equal, `str.len()==`            |         `str`         |    len_ne    |     str length not equal, `str.len()!=`      |         `str`         |
|    len_gt     |         str length great than, `str.len()>`          |         `str`         |    len_lt    |      str length less than, `str.len()<`      |         `str`         |
|    len_ge     |       str length great or equal, `str.len()>=`       |         `str`         |    len_le    |   str length less or equal, `str.len()<=`    |         `str`         |
|   contains    |         str contains, `str.contains(target)`         |         `str`         | not_contains | str not contains, `str.not_contains(target)` |         `str`         |
|      in       |              str in,  `str.in(target)`               |         `str`         |    not_in    |       str not in, `str.not_in(target)`       |         `str`         |
| lower_last_in | lower str last char in, `str.lower()[-1].in(target)` |         `str`         |              |                                              |                       |

#### 1.3.2 MD5 Filtering

Add an `md5` field under the `filter` section to initiate MD5 filtering. Multiple MD5 filtering criteria can be applied simultaneously, with the intersection of multiple conditions being taken. 

例如:
* `badcase.txt` is a list of MD5s, aiming to filter out the entries listed in these lists.
* `badcase.json` is a dictionary, where the key is the MD5 and the value is a text-related `tag`. The goal is to filter out specific `tags`.

```shell
filter:
    md5:
        - name: badcase1
          path:
            - badcase1.txt
          type: list
          action: in
          is_valid: false
        - name: badcase2
          path: badcase2.json
          type: dict
          action: eq
          target: 'Specified tag'
          is_valid: false
```

Each filtering criterion includes five fields:

|        Field Name         | Type  |                                Description                                |                                        Value Range                                        |
|:------------------:|:---:|:-----------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
|        name        | Required  |     The name of the filtering criterion, which can be customized for ease of statistics.                       |                                          -                                           |
|        path        | Required  | The path to the filtering file, which can be a single path or multiple paths provided in a list format. Supports `.txt`, `.json`, `.pkl` formats. |                                          -                                           |
|        type        | Required  |                         The type of records in the filtering file.                       |                                   `list` or `dict`                               | 
|       action       | Required  |                           The filtering action.                             | For `list`: `in`, `not_in`; For `dict`: `eq`, `ne`, `gt`, `lt`, `ge`, `le`|
|       target       | Optional  |                         The filtering criterion.                              |                               Required when type is `dict`                               |
|      is_valid      | Required  |                 Whether a hit on action+target is considered valid or invalid.                  |                                  `true` or `false`                           |
| arrow_file_keyword | Optional  |                       Keywords in the Arrow file path.                       |                                          -                                           |

### 1.4 Advanced Filtering

`idk` also supports some more advanced filtering functions.

#### 1.4.1  Filtering Criteria Applied to Part of Arrow Files

Using the `arrow_file_keyword` parameter, filtering criteria can be applied only to part of the Arrow files. 

For example:

* The filtering criterion `height>=0` applies only to arrows whose path includes `human`.
* The filtering criterion “keep samples in goodcase.txt” applies only to arrows whose path includes `human`.

```yaml
filter:
    column:
        - name: height
          type: int
          action: ge
          target: 512
          default: 1024
          arrow_file_keyword:
              - human

    md5:
        - name: goodcase
          path: goodcase.txt
          type: list
          action: in
          is_valid: true
          arrow_file_keyword:
            - human
```

#### 1.4.2 The “or” Logic in Filtering Criteria

By default, filtering criteria follow an “and” logic. If you want two or more filtering criteria to follow an “or” logic, you should use the `logical_or` field. The column filtering criteria listed under this field will be combined using an “or” logic.

```yaml
filter:  
    column:  
        - logical_or:  
            -   name: md5  
                type: str  
                action: lower_last_in  
                target: '02468ace'  
                default: ''  
            -   name: text_zh  
                type: str  
                action: contains  
                target: 'turtle|rabbit|duck|swan|peacock|hedgehog|wolf|fox|seal|crow|deer'  
                default: ''
```

Special Note: The `logical_or` field is applicable only to the `column` filtering criteria within `filter`.

1.4.3 Excluding Certain Arrows from the Source

While wildcards can be used to fetch multiple arrows at once, there might be instances where we want to exclude some of them. This can be achieved through the `exclude` field. Keywords listed under `exclude`, if found in the path of the current group of arrows, will result in those arrows being excluded.

```yaml
source:
    - /HunYuanDiT/dataset/porcelain/arrows/*.arrow:
        exclude:
            - arrow1
            - arrow2
```

### 1.5 Repeating Samples  
  
`idk` offers the capability to repeat either all samples or specific samples during dataset creation. There are three types of repeaters:  
* Directly repeating the source  
* Based on keywords in the Arrow file name (enable repeat conditions by adding the `repeater` field in the configuration file)  
* Based on an MD5 file (enable repeat conditions by adding the `repeater` field in the configuration file)  
  
**Special Note:** The above three conditions can be used simultaneously. If a sample meets multiple repeat conditions, the highest number of repeats will be taken.  
  
#### 1.5.1 Repeating the Source  
  
In the source, for the Arrow(s) you want to repeat (which can include wildcards like `*`, `?`), add `repeat: n` to mark the number of times to repeat.  
  
```yaml  
source:  
    - /HunYuanDiT/dataset/porcelain/arrows/*.arrow:  
        repeat: 10  
```

**Special Note: Add a colon at the end of the Arrow path**

#### 1.5.2 Arrow Keywords

Add an `arrow_file_keyword` field under the `repeater` section.

```yaml
repeater:  
    arrow_file_keyword:  
        - repeat: 8  
          keyword:  
            - Lolita anime style  
            - Minimalist style  
        - repeat: 5  
          keyword:  
              - Magical Barbie style  
              - Disney style  
```

Each repeat condition includes two fields:


|   Field Name	   | Type  |       Description        | Value Range |
|:-------:|:---:|:---------------:|:----:|
| repeat  | Required  |  The number of times to repeat  |  Number  |
| keyword | Required  | Keywords in the Arrow file path |  -    |


#### 1.5.3 MD5 File

Add an `md5` field under the `repeater` section.

```yaml
repeater:
    md5:
        - name: goodcase1
          path: /HunYuanDiT/dataset/porcelain/md5_repeat_1.json
          type: dict
          plus: 3
        - name: goodcase2
          path: /HunYuanDiT/dataset/porcelain/md5_repeat_2.json
          type: list
          repeat: 6
```

Each repeat condition includes the following fields:


|  Field Name   | Type  |                       Description	                     |            Value Range               |
|:------:|:---:|:----------------------------------------------:|:---------------------------:|
|  name  | Required  |               Custom name for the repeat condition                |              -              |
|  path  | Required  |                 Path to the file containing MD5s               | `.txt`, `.json`, or `.pkl` formats |
|  type  | Required  |                  Type of the MD5 file                 |       `list` or `dict`       |
| repeat | Optional  |     Number of times to repeat, this will override plus     |             Integer              |
|  plus  | Optional  | Adds this value to the value obtained from the MD5 file as the number of repeats |             Integer              |

### 1.6 Deduplication

To deduplicate, add `remove_md5_dup` and set it to `true`. For example:

```yaml
remove_md5_dup: true
```

**Special Note:** Deduplication is performed after repeat conditions, so using them together will make repeat ineffective.

### 1.7 Creating a Base Index V2 Dataset with Python Code

```python
from index_kits import IndexV2Builder

builder = IndexV2Builder(arrow_files)
builder.save('data_v2.json')
```



## 2. Creating a Multireso Index V2 Dataset

### 2.1 Creating a Multireso Index V2 Dataset with `idk`
To create a multi-resolution dataset, you can do so through a configuration file:

```shell
src:
    - /HunYuanDiT/dataset/porcelain/jsons/a.json
    - /HunYuanDiT/dataset/porcelain/jsons/b.json
    - /HunYuanDiT/dataset/porcelain/jsons/c.json
base_size: 512
reso_step: 32
min_size: 512
```

The fields are as follows:

|      Field Name      | Type  |                           Description                           |              Value Range               |
|:-------------:|:---:|:------------------------------------------------------:|:-------------------------------:|
|      src      | Required  |       Path(s) to the Base Index V2 file, can be single or multiple           |                -                |
|   base_size   | Required  |              The base resolution (n, n) from which to create multiple resolutions               |        Recommended values: 256/512/1024      |
|   reso_step   | Optional  | The step size for traversing multiple resolutions. Choose either this or target_ratios	 |          Recommended values: 16/32/64         |
| target_ratios | Optional  | Target aspect ratios, a list. Choose either this or reso_step	 | Recommended values: 1:1, 4:3, 3:4, 16:9, 9:16  |
|     align     | Optional  | When using target_ratios, the multiple to align the target resolution to	 | Recommended value: 16 (2x patchify, 8x VAE) |
|   min_size    | Optional  | The minimum resolution filter for samples when creating from Base to Multireso Index V2	 |       Recommended values: 256/512/1024      |
|   md5_file    | Optional  | A pre-calculated dictionary of image sizes in pkl format, key is MD5, value is (h, w)	 |                -                |


### 2.2 Creating a Multireso Index V2 Dataset with Python Code

First, import the necessary function

```python
from index_kits import build_multi_resolution_bucket  
  
md5_hw = None  
# If many arrows in your Index V2 lack height and width, you can pre-calculate them and pass through the md5_hw parameter.  
# md5_hw = {'c67be1d8f30fd0edcff6ac99b703879f': (720, 1280), ...}  
# with open('md5_hw.pkl', 'rb') as f:  
#     md5_hw = pickle.load(f)  
  
index_v2_file_path = 'data_v2.json'  
index_v2_save_path = 'data_multireso.json'  
  
# Method 1: Given base_size and reso_step, automatically calculate all resolution buckets.  
build_multi_resolution_bucket(base_size=1024,  
                              reso_step=64,  
                              min_size=1024,  
                              src_index_files=index_v2_file_path,   
                              save_file=index_v2_save_path,   
                              md5_hw=md5_hw)  
  
# Method 2: Given a series of target aspect ratios, automatically calculate all resolution buckets.  
build_multi_resolution_bucket(base_size=1024,  
                              target_ratios=["1:1", "3:4", "4:3", "16:9", "9:16"],  
                              align=16,  
                              min_size=1024,  
                              src_index_files=index_v2_file_path,   
                              save_file=index_v2_save_path,   
                              md5_hw=md5_hw)  
```

**Note：** If both reso_step and target_ratios are provided, target_ratios will be prioritized.

