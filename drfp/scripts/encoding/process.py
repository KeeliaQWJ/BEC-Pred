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
from sklearn.model_selection import train_test_split

def split_and_save_data(X, y, smiles, output_folder, random_seed):
    np.random.seed(random_seed)

    train_split = 0.8 
    eval_split = 0.1 
    test_split = 0.1 

    X_train, X_temp, y_train, y_temp, smiles_train, smiles_temp = train_test_split(
        X, y, smiles, test_size=(1 - train_split), random_state=random_seed)

    eval_size = eval_split / (1 - train_split)  
    X_eval, X_test, y_eval, y_test, smiles_eval, smiles_test = train_test_split(
        X_temp, y_temp, smiles_temp, test_size=eval_size, random_state=random_seed)

    train_filename = os.path.join(output_folder, f"train_{random_seed}.pkl")
    eval_filename = os.path.join(output_folder, f"eval_{random_seed}.pkl")
    test_filename = os.path.join(output_folder, f"test_{random_seed}.pkl")

    with open(train_filename, "wb+") as f:
        pickle.dump((X_train, y_train, smiles_train), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(eval_filename, "wb+") as f:
        pickle.dump((X_eval, y_eval, smiles_eval), f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(test_filename, "wb+") as f:
        pickle.dump((X_test, y_test, smiles_test), f, protocol=pickle.HIGHEST_PROTOCOL)


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
    )

    k, m = divmod(len(smiles), cpu_count)
    smiles_chunks = (
        smiles[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(cpu_count)
    )

    results = []
    with multiprocessing.Pool(cpu_count) as p:
        results = p.map(partial(encode, length=length, radius=radius), smiles_chunks)

    return np.array([item for s in results for item in s])


def add_split_to_filepath(filepath: str, split: str) -> str:
    name, ext = os.path.splitext(filepath)
    return f"{name}_{split}{ext}"

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

        print(f"{len(df)} reactions...")

        smiles = df[cols[0]].to_numpy()
        y = df[cols[1]].to_numpy()

        logger.info(f"generating drfp fingerprints ({s})")
        X = encode_dataset(smiles, length, radius)
        split_and_save_data(X, y, smiles, output_folder, random_seed)

    
if __name__ == "__main__":
    main()
