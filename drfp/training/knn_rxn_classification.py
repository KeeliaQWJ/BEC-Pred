# Adapted from: https://github.com/rxn4chemistry/rxnfp/blob/master/nbs/10_results_uspto_1k_tpl.ipynb
import os  
import pickle
import click
import faiss
import numpy as np
from pathlib import Path
from typing import Optional
from collections import Counter
from pycm import ConfusionMatrix
import sklearn.metrics
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def get_nearest_neighbours_prediction(
    train_X: np.array, train_y: np.array, eval_X: np.array, n_neighbours: int = 5
) -> list:
    """
    Use faiss to make a K-nearest neighbour prediction
    """
    # Indexing
    index = faiss.IndexFlatL2(len(train_X[0]))
    index.add(train_X.astype(np.float32))

    # Querying
    _, results = index.search(eval_X.astype(np.float32), n_neighbours)

    # Scoring
    y_pred = get_pred(train_y, results)

    return y_pred


def get_pred(y: list, results: list) -> list:
    """
    Get most common label from nearest neighbour list
    """
    y_pred = []
    for i, r in enumerate(results):
        y_pred.append(Counter(y[r]).most_common(1)[0][0])
    return y_pred


def get_cache_confusion_matrix(
    name: str, actual_vector: list, predict_vector: list
) -> ConfusionMatrix:
    """
    Make confusion matrix and save it.
    """
    cm_cached = load_confusion_matrix(f"{name}.pickle")

    if cm_cached is not None:
        return cm_cached

    cm = ConfusionMatrix(actual_vector=actual_vector, predict_vector=predict_vector)
    cm.save_html(name)
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(cm, f)
    return cm


def load_confusion_matrix(path: str) -> Optional[ConfusionMatrix]:
    """
    Load confusion matrix if existing.
    """
    if Path(path).is_file():
        return pickle.load(open(path, "rb"))
    return None

def f1_multiclass(y_true,y_pred):
      return sklearn.metrics.f1_score(y_true,y_pred, average='weighted')

def prec_multiclass(y_true,y_pred):
      return sklearn.metrics.precision_score(y_true,y_pred, average='weighted')

def rec_multiclass(y_true,y_pred):
      return sklearn.metrics.recall_score(y_true,y_pred, average='weighted')


@click.command()
@click.argument("input_train_filepath", type=click.Path(exists=True))
@click.argument("input_test_filepath", type=click.Path(exists=True))
@click.option("--cm-name", type=str, default="cm")
@click.option("--reduce", type=float, default=1.0)
def main(input_train_filepath, input_test_filepath, cm_name: str, reduce: float):
    # Initialize lists to store evaluation metrics over 5 iterations
    acc_list = []
    mcc_list = []
    f1_list = []

    for i in range(5):
        train_file = os.path.join(input_train_filepath, f'train_{i}.pkl')
        test_file = os.path.join(input_test_filepath, f'test_{i}.pkl')

        X_train, y_train, _ = pickle.load(open(train_file, "rb"))
        X_test, y_test, _ = pickle.load(open(test_file, "rb"))

        # Train a K-nearest neighbors classifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = [
        int(i) for i in get_nearest_neighbours_prediction(X_train, y_train, X_test)
        ]
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
        f1 = f1_multiclass(y_test, y_pred)

        # Append the metrics to the lists
        acc_list.append(acc)
        mcc_list.append(mcc)
        f1_list.append(f1)

    # Calculate mean, variance, and standard deviation for each metric
    mean_acc = np.mean(acc_list)
    var_acc = np.var(acc_list)
    std_acc = np.std(acc_list)

    mean_mcc = np.mean(mcc_list)
    var_mcc = np.var(mcc_list)
    std_mcc = np.std(mcc_list)

    mean_f1 = np.mean(f1_list)
    var_f1 = np.var(f1_list)
    std_f1 = np.std(f1_list)

    # Print the results
    print(f"Mean Accuracy: {mean_acc:.3f}, Variance: {var_acc:.3f}, Standard Deviation: {std_acc:.7f}")
    print(f"Mean MCC: {mean_mcc:.3f}, Variance: {var_mcc:.3f}, Standard Deviation: {std_mcc:.7f}")
    print(f"Mean F1 Score: {mean_f1:.3f}, Variance: {var_f1:.3f}, Standard Deviation: {std_f1:.7f}")


if __name__ == "__main__":
    main()
