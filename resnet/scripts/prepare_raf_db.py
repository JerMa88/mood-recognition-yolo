from __future__ import annotations

import argparse
import os
import shutil
import random

import kagglehub

random.seed(a=None) # change if repetiveness is desired
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def main(args: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Download and prepare RAF DB using kagglehub")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parsed = parser.parse_args(args)
    out_dir = parsed.out_dir
    val_ratio = parsed.val_ratio

    if os.path.isdir(out_dir):
        print(f"The directory '{out_dir}' exists. \nDownloading FER2013 into {out_dir} ... ")
    else:
        raise FileNotFoundError(f"The directory '{out_dir}' does not exist. Create this directory in your desired location manually please.")

    # kaggle.api.dataset_download_files(dataset='ananthu017/emotion-detection-fer', path=out_dir, unzip=True)

    path = kagglehub.dataset_download("dollyprajapati182/balanced-raf-db-dataset-7575-grayscale")
    
    if not os.path.isdir(os.path.join(out_dir, 'train')) or not os.path.isdir(os.path.join(out_dir, 'test')):
        print("Path to dataset files:", path, '\nNow moving to designated data folder:', out_dir)
        shutil.move(os.path.join(path, 'train'), out_dir)
        shutil.move(os.path.join(path, 'test'), out_dir)
        shutil.move(os.path.join(path, 'val'), out_dir)
    
    # if not os.path.isdir(os.path.join(out_dir, 'val')): 
    #     os.mkdir(os.path.join(out_dir, 'val'))
    
    # print(f'Splitting train into {val_ratio}/{1-val_ratio} for train/validation ...')
    # for emotion in EMOTIONS:
    #     train_dir = os.path.join(out_dir, 'train', emotion)
    #     val_dir = os.path.join(out_dir, 'val', emotion)
        
    #     if not os.path.isdir(val_dir): 
    #         os.mkdir(val_dir)
    #     images = os.listdir(train_dir)
    #     val_images = random.sample(images, k=int(val_ratio*len(images)))
        
    #     print(f'{int(val_ratio*len(images))} images moving into {val_dir} ...')
    #     try:
    #         for val_image in val_images:
    #             shutil.move(os.path.join(train_dir, val_image), os.path.join(val_dir, val_image))
    #     except Exception as e:print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

