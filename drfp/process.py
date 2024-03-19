# -*- coding: utf-8 -*-
import os
import pickle
import click
import logging
import multiprocessing
from functools import partial
from typing import Iterable

from drfp import DrfpEncoder

import numpy as np
import pandas as pd


def encode(smiles: Iterable, length: int = 1024, radius: int = 3) -> np.ndarray:
    return DrfpEncoder.encode(
        smiles,
        n_folded_length=length,
        radius=radius,
        rings=True,
    )


def encode_dataset(smiles: Iterable, length: int, radius: int) -> np.ndarray:
    """Encode the reaction SMILES to drfp"""

    cpu_count = (
        multiprocessing.cpu_count()
    )  # Data gets too big for piping when splitting less in python < 2.8

    # Split reaction SMILES for multiprocessing
    k, m = divmod(len(smiles), cpu_count)
    smiles_chunks = (
        smiles[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(cpu_count)
    )

    # Run the fingerprint generation in parallel
    results = []
    with multiprocessing.Pool(cpu_count) as p:
        results = p.map(partial(encode, length=length, radius=radius), smiles_chunks)

    return np.array([item for s in results for item in s])


def add_split_to_filepath(filepath: str, split: str) -> str:
    name, ext = os.path.splitext(filepath)
    return f"{name}_{split}{ext}"

def split_and_save_data(X, y, smiles, output_folder, random_seed):
    np.random.seed(random_seed)

    train_split = 0.8  # 80% 训练集
    eval_split = 0.1  # 10% 验证集
    test_split = 0.1   # 10% 测试集

    total_len = len(X)
    train_len = int(total_len * train_split)
    eval_len = int(total_len * eval_split)

    # 划分数据集
    X_train = X[:train_len]
    y_train = y[:train_len]
    smiles_train = smiles[:train_len]

    X_eval = X[train_len:train_len + eval_len]
    y_eval = y[train_len:train_len + eval_len]
    smiles_eval = smiles[train_len:train_len + eval_len]

    X_test = X[train_len + eval_len:]
    y_test = y[train_len + eval_len:]
    smiles_test = smiles[train_len + eval_len:]

    # 构建文件名
    train_filename = os.path.join(output_folder, f"train_{random_seed}.pkl")
    eval_filename = os.path.join(output_folder, f"eval_{random_seed}.pkl")
    test_filename = os.path.join(output_folder, f"test_{random_seed}.pkl")

    # 保存为pickle文件
    with open(train_filename, "wb+") as f:
        pickle.dump((X_train, y_train, smiles_train), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(eval_filename, "wb+") as f:
        pickle.dump((X_eval, y_eval, smiles_eval), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(test_filename, "wb+") as f:
        pickle.dump((X_test, y_test, smiles_test), f, protocol=pickle.HIGHEST_PROTOCOL)

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
@click.option("--cols", nargs=3, type=str, required=True)
@click.option("--sep", type=str, default="\t")
@click.option("--length", type=int, default=1024)
@click.option("--radius", type=int, default=3)
def main(input_filepath, output_folder, cols, sep, length, radius):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    df = pd.read_csv(input_filepath, sep=sep, usecols=list(cols))

    for s in range(5):
        random_seed = int(s)

        if os.path.exists(os.path.join(output_folder, f"train_{random_seed}.pkl")):
            print(f"Split {s} already exists.")
            continue
        # split_filepath = os.path.join(output_folder, add_split_to_filepath(s))

        # if os.path.exists(split_filepath):
        #     print(f"{split_filepath} already exists.")
        #     continue

        print(f"{len(df)} reactions...")

        smiles = df[cols[0]].to_numpy()
        y = df[cols[1]].to_numpy()

        logger.info(f"generating drfp fingerprints ({s})")
        X = encode_dataset(smiles, length, radius)
        split_and_save_data(X, y, smiles, output_folder, random_seed)

    
if __name__ == "__main__":
    main()
