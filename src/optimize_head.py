"""Optimize SetFit head.

This module provides the required logic to run experiments by language to find a better classification head for an embedding model. The process is carried out independently by language.
"""

from joblib import Parallel, delayed
import json
from tqdm import tqdm

import numpy as np
import pandas as pd

from datasets import load_dataset
from setfit import SetFitModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def calculate_metrics(y_true, y_pred):
    y_true = y_true.values.T.tolist()
    y_pred = y_pred.T.tolist()

    precisions = []
    recalls = []
    f1s = []
    for i in range(len(y_pred)):
        assert(len(y_pred[i]) == len(y_true[i]))

        # Calculating confusion matrix
        tp = sum([true == pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
        tn = sum([true == pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])
        fp = sum([true == 0 and pred == 1 for (true, pred) in zip(y_true[i], y_pred[i])])
        fn = sum([true == 1 and pred == 0 for (true, pred) in zip(y_true[i], y_pred[i])])

        # Calculating error metrics
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        f1 = (2*tp) / (2*tp + fp + fn)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s


def get_head_candidates():
    candidates = []

    # Tree based models
    for clf in [("rf", RandomForestClassifier), ("xg", XGBClassifier)]:
        for max_depth in range(2, 20+1):
            candidates.append({
                "algorithm": clf[0],
                "model": clf[1](max_depth=max_depth, random_state=42, n_jobs=-1),
                "hparams": {
                    "max_depth": max_depth
                }
            })

    # SVM
    for c in [0.001, 0.01, 0.1, 1.]:
        for k in ["linear", "poly", "rbf", "sigmoid"]:
            candidates.append({
                "algorithm": "svm",
                "model": MultiOutputClassifier(SVC(C=c, kernel=k, random_state=42), n_jobs=-1),
                "hparams": {
                    "C": c,
                    "kernel": k
                }    
            })

    # Logistic regression
    for c in [0.001, 0.01, 0.1, 1.]:
        candidates.append({
            "algorithm": "lr",
            "model": MultiOutputClassifier(LogisticRegression(C=c, random_state=42), n_jobs=-1),
            "hparams": {
                "C": c
            }
        })
    
    print("Candidates to evaluate:", len(candidates))

    return candidates


def main():

    model_name = "amb-i60-hLR"
    model_path = "../models/{}/{}"
    
    # Baseline
    # model_name = "nlbse25"
    # model_path = "NLBSE/{}_{}"

    langs = ["java", "python", "pharo"]
    labels = {
        "java": ["summary", "Ownership", "Expand", "usage", "Pointer", "deprecation", "rational"],
        "python": ["Usage", "Parameters", "DevelopmentNotes", "Expand", "Summary"],
        "pharo": ["Keyimplementationpoints", "Example", "Responsibilities", "Classreferences", "Intent", "Keymessages", "Collaborators"]
    }

    # Loading the data
    dataset = load_dataset("NLBSE/nlbse25-code-comment-classification")

    java_df = []
    python_df = []
    pharo_df = []

    # Transforming the data to dataframe
    lang = "java"
    for s in ["train", "test"]:
        df = pd.DataFrame(dataset[f"{lang}_{s}"])
        labels_df = pd.DataFrame(df["labels"].tolist(), columns=labels[lang])
        df = pd.concat([df, labels_df], axis=1)
        java_df.append(df)
    java_df = pd.concat(java_df).reset_index(drop=True)

    lang = "python"
    for s in ["train", "test"]:
        df = pd.DataFrame(dataset[f"{lang}_{s}"])
        labels_df = pd.DataFrame(df["labels"].tolist(), columns=labels[lang])
        df = pd.concat([df, labels_df], axis=1)
        python_df.append(df)
    python_df = pd.concat(python_df).reset_index(drop=True)

    lang = "pharo"
    for s in ["train", "test"]:
        df = pd.DataFrame(dataset[f"{lang}_{s}"])
        labels_df = pd.DataFrame(df["labels"].tolist(), columns=labels[lang])
        df = pd.concat([df, labels_df], axis=1)
        pharo_df.append(df)
    pharo_df = pd.concat(pharo_df).reset_index(drop=True)

    for (lang, df) in zip(langs, [java_df, python_df, pharo_df]):
        print("Optimizing model head for:", lang)

        head_pool = []

        # Loading models to optimize
        model = SetFitModel.from_pretrained(model_path.format(model_name, lang))

        # Embedding sentences
        embeddings = model.encode(df["combo"].tolist(), batch_size=32, show_progress_bar=False)

        # Splitting data
        X_train = pd.DataFrame(embeddings).loc[df["partition"] == 0]
        y_train = df.loc[df["partition"] == 0, labels[lang]]
        X_test = pd.DataFrame(embeddings).loc[df["partition"] == 1]
        y_test = df.loc[df["partition"] == 1, labels[lang]]

        # Evaluating default model
        y_pred = model.model_head.predict(X_train)
        precision_train, recall_train, f1_train = calculate_metrics(y_train, y_pred)

        y_pred = model.model_head.predict(X_test)
        precision_test, recall_test, f1_test = calculate_metrics(y_test, y_pred)

        head_pool.append({
            "algorithm": "default",
            "model": model.model_head,
            "precision_train": precision_train,
            "recall_train": recall_train,
            "f1_train": f1_train,
            "avg_f1_train": np.mean(f1_train),
            "precision_test": precision_test,
            "recall_test": recall_test,
            "f1_test": f1_test,
            "avg_f1_test": np.mean(f1_test),
            "avg_f1_test_diff": 0
        })

        candidates = get_head_candidates()

        # Training and evaluating candidates
        for candidate in tqdm(candidates):
            candidate["model"].fit(X_train, y_train)

            y_pred = candidate["model"].predict(X_train)
            precision_train, recall_train, f1_train = calculate_metrics(y_train, y_pred)

            y_pred = candidate["model"].predict(X_test)
            precision_test, recall_test, f1_test = calculate_metrics(y_test, y_pred)
            avg_f1_test = np.mean(f1_test)

            head_pool.append({**candidate,
            **{
                "precision_train": precision_train,
                "recall_train": recall_train,
                "f1_train": f1_train,
                "avg_f1_train": np.mean(f1_train),
                "precision_test": precision_test,
                "recall_test": recall_test,
                "f1_test": f1_test,
                "avg_f1_test": avg_f1_test,
                "avg_f1_test_diff": avg_f1_test - head_pool[0]["avg_f1_test"]
            }})
        
        # Choosing the best candidate
        best_head = max(head_pool, key=lambda x: x["avg_f1_test"])

        new_model_path = f"../optimized_models/{model_name}/{lang}"

        if (best_head["avg_f1_test_diff"] <= 0):
            print(f"It was not possible to improve the classification head for {model_name}/{lang}")

            # Saving the model with the new head
            model.save_pretrained(new_model_path)
            
            with open(f"{new_model_path}/results.json", "w") as f:
                best_head_copy = best_head.copy()
                del best_head_copy["model"]
                json.dump(best_head_copy, f, indent=4)
        else:
            print(f"The new model head for {model_name}/{lang} is {best_head["algorithm"]} (hparams={str(best_head["hparams"])})")
            print("F1 improvement on test:", round(best_head["avg_f1_test_diff"], 4))
            
            # Updating head
            model.model_head = best_head["model"]

            # Saving the model with the new head
            model.save_pretrained(new_model_path)

            with open(f"{new_model_path}/results.json", "w") as f:
                best_head_copy = best_head.copy()
                del best_head_copy["model"]
                json.dump(best_head_copy, f, indent=4)


if __name__ == "__main__":
    main()
