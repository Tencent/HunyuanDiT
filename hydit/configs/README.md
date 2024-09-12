# 优化混元DiT模型启动配置流程

## 配置目录结构

参考[MMEngine/config](https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/config.py)的配置风格和代码优化了混元DiT模型的启动配置流程，将配置参数按照数据、模型和启动流程划分，使用py文件配置模型参数；在需要新增配置文件时，可引用默认配置

新增`hydit/configs`目录用于存储启动配置文件，目录结构如下：
```bash
- configs
    - base # 默认配置
        - dataset   # 数据集配置
        - model     # 模型配置
        - schedule  # 启动配置
    - train # 基于默认配置文件的训练配置文件
```

## 启动流程

在加载配置时，为了保留原有的代码结构，新增配置加载文件`hydit/config_engine.py`，在`train_deepspeed.py`中仅修改了函数`get_args`的引用模块

```python
# 修改前 from hydit.config import get_args
from hydit.config_engine import get_args
```

由于全参数训练和仅训练Lora都使用的deepspeed，所以新增`train_deepspeed.sh`脚本启动训练，启动命令如下：

```bash
PYTHONPATH=./ sh hydit/train_deepspeed.sh --config hydit/configs/train/train_lora_dit_g2_1024p_single.py 
```

其中，`config`参数传递的为训练配置文件相对路径