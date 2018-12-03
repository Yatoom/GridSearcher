import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.style.use('seaborn')

frame = pd.read_csv("../results/result.csv", index_col=0)
parameters = [
    "learning_rate", "max_depth", "min_child_samples", "num_leaves", "reg_alpha", "reg_lambda"
]

optima_per_task = frame.groupby("task_id").max()["acc"]
tasks = np.array(frame["task_id"])
optima = optima_per_task[tasks].reset_index(drop=True)
frame["percent"] = frame["acc"] / optima
mean_perc = frame[frame["max_depth"] == 12].groupby("num_leaves").mean()["percent"]
var_perc = frame[frame["max_depth"] == 12].groupby("num_leaves").var()["percent"]
plt.plot(mean_perc, marker='o')
plt.fill_between(var_perc.index, mean_perc - var_perc, mean_perc + var_perc, alpha=0.1)
plt.xlabel("Number of leaves")
plt.ylabel("Accuracy / optimum")
plt.show()

mean_time = frame[frame["max_depth"] == 12].groupby("num_leaves").mean()["eval_time"]
var_time = frame[frame["max_depth"] == 12].groupby("num_leaves").var()["eval_time"]
plt.plot(mean_time, marker='o')
plt.fill_between(var_time.index, mean_time - var_time, mean_time + var_time, alpha=0.1)
plt.xlabel("Number of leaves")
plt.ylabel("Running time")
plt.show()




# correlation_with_accuracy = frame.corr()["acc"][parameters].sort_values()
# correlation_with_time = frame.corr()["eval_time"][parameters].sort_values()
# correlation_with_accuracy.plot.barh(title="Correlation with accuracy", color="#444444")
# plt.show()
# correlation_with_time.plot.barh(title="Correlation with evaluation time", color="#444444")
# plt.show()