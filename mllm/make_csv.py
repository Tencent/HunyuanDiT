import os
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="mllm/images")
    parser.add_argument("--input_file", type=str, default="mllm/images/demo.csv")
    args = parser.parse_args()

    df = pd.DataFrame(columns=["img_path"])
    df["img_path"] = [
        os.path.join(args.img_dir, fn)
        for fn in os.listdir(args.img_dir)
        if fn.endswith(".png")
    ]

    df.to_csv(args.input_file, index=False)
    print("csv file saved to: ", args.input_file)
