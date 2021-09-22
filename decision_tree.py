from sklearn.tree import DecisionTreeClassifier
from azureml.core import Dataset
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Dataset


ds = Dataset.Tabular.from_delimited_files(path="https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv")
df = ds.to_pandas_dataframe(ds)

def clean(data):
    df = data
    df["age"] = df.age.apply(lambda s: 60 if s == 60.667 else s)
    df.drop(df[df["platelets"] == 263358.03].index, inplace=True)
    return df
df = clean(df)
x = df
y = df.pop("DEATH_EVENT")

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.20)
run = Run.get_context()


def main():
   
    parser = argparse.ArgumentParser()

    parser.add_argument('--criterion', type=str, default="gini", help="function to measure the quality of the split")
    parser.add_argument('--splitter', type=str, default="best", help="strategy used to split at each node")
    parser.add_argument('--max_depth', type=int, default=20, help="max depth of a tree")
    args = parser.parse_args()

    run.log("Criterion:", np.str(args.criterion))
    run.log("Splitter:", np.str(args.splitter))
    run.log("Max_depth:", np.int(args.max_depth))

    model = DecisionTreeClassifier(criterion=args.criterion, splitter=args.splitter, max_depth=args.max_depth).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrive-model.joblib') 

if __name__ == '__main__':
    main()

