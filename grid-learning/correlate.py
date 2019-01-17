import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

frame = pd.read_csv("../results/result.csv", index_col=0)
frame = frame.sort_values(["task_id", "n_estimators", "num_leaves", "max_depth", "min_child_samples", "learning_rate", "reg_alpha", "reg_lambda"])

tasks = np.unique(frame["task_id"])
tasks = np.delete(tasks, 0)  # Delete task 3
result = pd.DataFrame()
a, b = np.array(list(itertools.permutations(tasks, 2))).T
result["task_1"] = a
result["task_2"] = b


correlation = []
for index, item in enumerate(result.to_dict(orient="records")):
    task_1 = item["task_1"]
    task_2 = item["task_2"]
    corr, p_val = kendalltau(frame[frame["task_id"] == task_1], frame[frame["task_id"] == task_2])
    correlation.append(corr)

result["correlation"] = correlation

# frame = frame.groupby("task_id").rank()
print()
# frame.sort_values(["group"])