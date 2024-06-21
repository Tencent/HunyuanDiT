# -*- coding: utf-8 -*-
import datetime
import gc
import os
import time
from multiprocessing import Pool
import subprocess
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
import hashlib
from PIL import Image
import sys


def parse_data(data):
    try:
        img_path = data[0]

        with open(img_path, "rb") as fp:
            image = fp.read()
            md5 = hashlib.md5(image).hexdigest()
            
        with Image.open(img_path) as f:
            width, height = f.size

        return [data[1], md5, width, height, image]

    except Exception as e:
        print(f'error: {e}')
        return

def make_arrow(csv_root, dataset_root, start_id=0, end_id=-1):
    print(csv_root)
    arrow_dir = dataset_root
    print(arrow_dir)

    if not os.path.exists(arrow_dir):
        os.makedirs(arrow_dir)

    data = pd.read_csv(csv_root)
    data = data[["image_path", "text_zh"]]
    columns_list = data.columns.tolist()
    columns_list.append("image")

    if end_id < 0:
        end_id = len(data)
    print(f'start_id:{start_id}  end_id:{end_id}')
    data = data[start_id:end_id]
    num_slice = 5000
    start_sub = int(start_id / num_slice)
    sub_len = int(len(data) // num_slice)  # if int(len(data) // num_slice) else 1
    subs = list(range(sub_len + 1))
    for sub in tqdm(subs):
        arrow_path = os.path.join(arrow_dir, '{}.arrow'.format(str(sub + start_sub).zfill(5)))
        if os.path.exists(arrow_path):
            continue
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} start {sub + start_sub}")

        sub_data = data[sub * num_slice: (sub + 1) * num_slice].values

        bs = pool.map(parse_data, sub_data)
        bs = [b for b in bs if b]
        print(f'length of this arrow:{len(bs)}')

        columns_list = ["text_zh", "md5", "width", "height", "image"]
        dataframe = pd.DataFrame(bs, columns=columns_list)
        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(arrow_path, "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        del dataframe
        del table
        del bs
        gc.collect()


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python hydit/data_loader/csv2arrow.py ${csv_root} ${output_arrow_data_path} ${pool_num}")
        print("csv_root: The path to your created CSV file. For more details, see https://github.com/Tencent/HunyuanDiT?tab=readme-ov-file#truck-training")
        print("output_arrow_data_path: The path for storing the created Arrow file")
        print("pool_num: The number of processes, used for multiprocessing. If you encounter memory issues, you can set pool_num to 1")
        sys.exit(1)
    csv_root = sys.argv[1]
    output_arrow_data_path = sys.argv[2]
    pool_num = int(sys.argv[3])
    pool = Pool(pool_num)
    make_arrow(csv_root, output_arrow_data_path)
