import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure()
# fig.set_figheight(9)
# fig.set_figwidth(16)

# methods = ["rfr-rfr", "rfr-gbm", "rfr-rfr-interleaved", "rfr-gbm-interleaved", ]
methods = ["rfr-gbm-interleaved", "gbqr", "gbqr-median", "rfr-interleaved"]
xmax = []
for method in methods:
    y = pd.read_csv(f"../sim/1000-{method}-scores.csv", index_col=0)
    x = pd.read_csv(f"../sim/1000-{method}-eval-time.csv", index_col=0)
    std_y = y.std(axis=1)
    mean_y = y.mean(axis=1)
    mean_x = np.mean(np.cumsum(x), axis=1)
    plt.plot(mean_x, mean_y, label=method)
    plt.fill_between(mean_x, mean_y - std_y, mean_y + std_y, alpha=0.1)
    xmax.append(np.max(mean_x))
plt.ylim(0.996, 1.002)
plt.xlim(0, np.min(xmax))
plt.legend(loc="lower right")
plt.xlabel("Accumulated average time")
plt.ylabel("Accumulated maximum as fraction of optimum")
plt.show()

for method in methods:
    y = pd.read_csv(f"../sim/1000-{method}-scores.csv", index_col=0)
    x = pd.read_csv(f"../sim/1000-{method}-eval-time.csv", index_col=0)
    its = np.arange(0, len(y))
    mean_x = np.mean(np.cumsum(x), axis=1)
    std_x = np.std(np.cumsum(x), axis=1)
    plt.plot(its, mean_x, label=method)
    plt.fill_between(its, mean_x - std_x, mean_x + std_x, alpha=0.1)


plt.legend(loc="upper left")
plt.xlabel("Iteration")
plt.ylabel("Average time to reach iteration")
plt.show()
