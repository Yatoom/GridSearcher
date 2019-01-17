import os
import time
from random import shuffle

from arbok import ConditionalImputer
from lightgbm import LGBMClassifier
import openml
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import json
import sys
import connection
import tasks

# Connect to database

client, db = connection.connect()

# Gather Experiments
print("[SEARCH] Preparing experiments")
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
filename = os.path.join(dname, "light-rf.json")
with open(filename) as f:
    grid = json.load(f)

experiments = []
keys, values = zip(*grid.items())
for v in itertools.product(*values):
    experiments.append(dict(zip(keys, v)))
shuffle(experiments)

for task_id in tasks.tasks:
    task_ids = db.distinct("task_id")
    if task_id in task_ids:
        continue
    try:
        print("[SEARCH] Downloading task")
        # task_id = int(sys.argv[1])
        task = openml.tasks.get_task(task_id)

        print("[SEARCH] Downloading data")
        X, y = task.get_X_and_y()

        print("[SEARCH] Downloading splits")
        splits = task.download_split().split

        print("[SEARCH] Starting experiments")
        for experiment in tqdm(experiments, ascii=True):
            # model = LGBMClassifier(verbose=-1, **experiment)
            # model = RandomForestClassifier(n_jobs=-1, **experiment)
            model = LGBMClassifier(boosting_type="rf", verbose=-1, **experiment)
            accuracies = []
            kappas = []
            # f1_scores = []
            times = []

            for folds in splits.values():
                for fold in folds.values():
                    for repeat in fold.values():
                        start = time.time()
                        train_index, test_index = repeat.train, repeat.test
                        model.fit(X[train_index], y[train_index])
                        y_pred = model.predict(X[test_index])
                        end = time.time()

                        times.append(end - start)
                        accuracies.append(accuracy_score(y[test_index], y_pred))
                        kappas.append(cohen_kappa_score(y[test_index], y_pred))
                        # f1_scores.append(f1_score(y[test_index], y_pred))

            eval_time = np.sum(times)
            mean_acc = np.mean(accuracies)
            # mean_f1 = np.mean(f1_scores)
            mean_kappa = np.mean(kappas)

            db.insert({
                "task_id": task_id,
                **experiment,
                "eval_time": eval_time,
                "acc": mean_acc,
                # "f1": mean_f1,
                "kappa": mean_kappa
            })
    except:
        print("oops")

