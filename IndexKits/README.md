# IndexKits  
  
[TOC]  
  
## Introduction  
  
Index Kits (`index_kits`) for streaming Arrow data.  
  
* Supports creating datasets from configuration files.  
* Supports creating index v2 from arrows.  
* Supports the creation, reading, and usage of index files.  
* Supports the creation, reading, and usage of multi-bucket, multi-resolution indexes.  
  
`index_kits` has the following advantages:  
  
* Optimizes memory usage for streaming reads, supporting the reading of data at the billion level.  
* Optimizes memory usage and reading speed of Index files.  
* Supports the creation of Base/Multireso Index V2 datasets, including data filtering, repeating, and deduplication during creation.  
* Supports loading various dataset types (Arrow files, Base Index V2, multiple Base Index V2, Multireso Index V2, multiple Multireso Index V2).  
* Uses a unified API interface to access images, text, and other attributes for all dataset types.  
* Built-in shuffle, `resize_and_crop`, and returns the coordinates of the crop.  
  
## Installation  
  
* **Install from pre-compiled whl (recommended)**  
  
* Install from source  
  
  ```shell  
  cd IndexKits  
  pip install -e .  
  ```

## Usage

### Loading an Index V2 File

```python
from index_kits import ArrowIndexV2  
  
index_manager = ArrowIndexV2('data.json')  
  
# You can shuffle the dataset using a seed. If a seed is provided (i.e., the seed is not None), this shuffle will not affect any random state.  
index_manager.shuffle(1234, fast=True)  
  
for i in range(len(index_manager)):  
    # Get an image (requires the arrow to contain an image/binary column):  
    pil_image = index_manager.get_image(i)  
    # Get text (requires the arrow to contain a text_zh column):  
    text = index_manager.get_attribute(i, column='text_zh')  
    # Get MD5 (requires the arrow to contain an md5 column):  
    md5 = index_manager.get_md5(i)  
    # Get any attribute by specifying the column (must be contained in the arrow):  
    ocr_num = index_manager.get_attribute(i, column='ocr_num')  
    # Get multiple attributes at once  
    item = index_manager.get_data(i, columns=['text_zh', 'md5'])     # i: in-dataset index  
    print(item)  
    # {  
    #      'index': 3,              # in-json index  
    #      'in_arrow_index': 3,     # in-arrow index  
    #      'arrow_name': '/HunYuanDiT/dataset/porcelain/00000.arrow',   
    #      'text_zh': 'Fortune arrives with the auspicious green porcelain tea cup',   
    #      'md5': '1db68f8c0d4e95f97009d65fdf7b441c'  
    # }  
```


### Loading a Set of Arrow Files

If you have a small batch of arrow files, you can directly use `IndexV2Builder` to load these arrow files without creating an Index V2 file.

```python
from index_kits import IndexV2Builder  

index_manager = IndexV2Builder(arrow_files).to_index_v2()  
```

### Loading a Multi-Bucket (Multi-Resolution) Index V2 File

When using a multi-bucket (multi-resolution) index file, refer to the following example code, especially the definition of SimpleDataset and the usage of MultiResolutionBucketIndexV2.

```python
from torch.utils.data import DataLoader, Dataset  
import torchvision.transforms as T  
  
from index_kits import MultiResolutionBucketIndexV2  
from index_kits.sampler import BlockDistributedSampler  
  
class SimpleDataset(Dataset):  
    def __init__(self, index_file, batch_size, world_size):  
        # When using multi-bucket (multi-resolution), batch_size and world_size need to be specified for underlying data alignment.  
        self.index_manager = MultiResolutionBucketIndexV2(index_file, batch_size, world_size)  
  
        self.flip_norm = T.Compose(  
            [  
                T.RandomHorizontalFlip(),  
                T.ToTensor(),  
                T.Normalize([0.5], [0.5]),  
            ]  
        )  
  
    def shuffle(self, seed, fast=False):  
        self.index_manager.shuffle(seed, fast=fast)  
  
    def __len__(self):  
        return len(self.index_manager)  
  
    def __getitem__(self, idx):  
        image = self.index_manager.get_image(idx)  
        original_size = image.size    # (w, h)  
        target_size = self.index_manager.get_target_size(idx)     # (w, h)  
        image, crops_coords_left_top = self.index_manager.resize_and_crop(image, target_size)  
        image = self.flip_norm(image)  
        text = self.index_manager.get_attribute(idx, column='text_zh')  
        return image, text, (original_size, target_size, crops_coords_left_top)  
  
batch_size = 8      # batch_size per GPU  
world_size = 8      # total number of GPUs  
rank = 0            # rank of the current process  
num_workers = 4     # customizable based on actual conditions  
shuffle = False     # must be set to False  
drop_last = True    # must be set to True  
  
# Correct batch_size and world_size must be passed in, otherwise, the multi-resolution data cannot be aligned correctly.   
dataset = SimpleDataset('data_multireso.json', batch_size=batch_size, world_size=world_size)  
# Must use BlockDistributedSampler to ensure the samples in a batch have the same resolution.  
sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=rank,  
                                  shuffle=shuffle, drop_last=drop_last, batch_size=batch_size)  
loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,  
                    num_workers=num_workers, pin_memory=True, drop_last=drop_last)  
  
for epoch in range(10):  
   # Please use the shuffle method provided by the dataset, not the DataLoader's shuffle parameter.  
   dataset.shuffle(epoch, fast=True)  
   for batch in loader:  
       pass  
```


## Fast Shuffle

When your index v2 file contains hundreds of millions of samples, using the default shuffle can be quite slow. Therefore, it’s recommended to enable `fast=True` mode:

```python
index_manager.shuffle(seed=1234, fast=True)
```

This will globally shuffle without keeping the indices of the same arrow together. Although this may reduce reading speed, it requires a trade-off based on the model’s forward time.