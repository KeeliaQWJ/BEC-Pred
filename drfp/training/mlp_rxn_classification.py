import os
import pickle
import click
import numpy as np
import sklearn.metrics
from pathlib import Path
from typing import Optional
from collections import Counter
from pycm import ConfusionMatrix
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x



def get_pred(train_X, train_y, eval_X, n_classes):
    input_dim = len(train_X[0])
    hidden_dim = 2048

    model = SimpleMLP(input_dim, hidden_dim, n_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_X = torch.Tensor(train_X)
    train_y = torch.LongTensor(train_y)

    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/500], Loss: {loss.item()}")


    eval_X = torch.Tensor(eval_X)
    outputs = model(eval_X)
    _, predictions = torch.max(outputs, 1)

    return predictions.numpy()

def f1_multiclass(y_true,y_pred):
      return sklearn.metrics.f1_score(y_true,y_pred, average='weighted')

def prec_multiclass(y_true,y_pred):
      return sklearn.metrics.precision_score(y_true,y_pred, average='weighted')

def rec_multiclass(y_true,y_pred):
      return sklearn.metrics.recall_score(y_true,y_pred, average='weighted')


def get_cache_confusion_matrix(
    name: str, actual_vector: list, predict_vector: list
) -> ConfusionMatrix:
    """
    Make confusion matrix and save it.
    """
    cm = ConfusionMatrix(actual_vector=actual_vector, predict_vector=predict_vector)
    cm.save_html(name)
    with open(f"{name}.pickle", "wb") as f:
        pickle.dump(cm, f)
    return cm

n_classes = 308
@click.command()
@click.argument("input_train_filepath", type=click.Path(exists=True))
@click.argument("input_test_filepath", type=click.Path(exists=True))
def main(input_train_filepath, input_test_filepath):
    
    acc_list = []
    mcc_list = []
    f1_list = []

    for i in range(5):
        train_file = os.path.join(input_train_filepath, f'train_{i}.pkl')
        test_file = os.path.join(input_test_filepath, f'test_{i}.pkl')

        X_train, y_train, _ = pickle.load(open(train_file, "rb"))
        X_test, y_test, _ = pickle.load(open(test_file, "rb"))
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        #n_classes = len(le.classes_)
        
        print(y_train)
        y_pred = get_pred(X_train, y_train, X_test, n_classes)

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
