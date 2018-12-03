import time

import numpy as np
import pandas as pd
# Settings
from lightgbm import LGBMRegressor
from scipy.stats import norm
from smac.epm.rf_with_instances import RandomForestWithInstances
# Load data
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

frame = pd.read_csv("../results/result.csv", index_col=0)

# Drop duplicates, if any
parameters = [
    "learning_rate", "max_depth", "min_child_samples", "n_estimators", "num_leaves", "reg_alpha", "reg_lambda"
]
frame.drop_duplicates(subset=parameters)

# Get X and y
X = frame[parameters]
Y = frame["acc"]
groups = frame["task_id"]
duration = frame["eval_time"]
X, Y, groups, duration = np.array(X), np.array(Y), np.array(groups), np.array(duration)
unique_groups = np.unique(groups)


# Wrapper
class Wrap(RandomForestWithInstances):
    def __init__(self, types, bounds, *args, **kwargs):
        super().__init__(types=types, bounds=bounds, *args, **kwargs)

    def fit(self, X, y):
        return self.train(X, y)

    def predict(self, X: np.ndarray):
        means, vars = super().predict(X)
        return means

    def predict_mean_variance(self, X: np.ndarray):
        return super().predict(X)

    def get_params(self, deep=False):
        return {
            "types": self.types,
            "bounds": self.bounds,
        }

    def set_params(self, deep=False, **kwargs):
        pass


# Setup LightGBM model
lgbm_time_estimator = LGBMRegressor(verbose=-1, min_child_samples=1, objective="mse", num_leaves=4, reg_alpha=0.20,
                                    learning_rate=0.15, min_data_in_bin=1)
lgbm_score_estimator = LGBMRegressor(verbose=-1, min_child_samples=1, objective="quantile", num_leaves=8, alpha=0.90,
                                     min_data_in_bin=1)

# Setup RFR
bounds_ = list(zip(X.min(axis=0), X.max(axis=0)))
types_ = [0 for _ in range(len(bounds_))]
rfr_score_estimator = Wrap(np.array(types_), np.array(bounds_))
rfr_time_estimator = Wrap(np.array(types_), np.array(bounds_))

startup_rounds = 3
no_random_sampling = True
for mode in ["rfr-rfr", "rfr-gbm"]:  # "gbqr", "rfr", "rfr-rfr",

    # Predict EI
    def predict_ei(points):
        if mode == "gbqr":
            ei = lgbm_score_estimator.predict(points)
            ei[observed] = -1
            return ei
        elif mode == "gbqr-median":
            upper = lgbm_score_estimator.predict(points)
            time = lgbm_time_estimator.predict(points)
            eips = (upper - np.median(upper)) / np.maximum(0, time)
            eips[observed] = -1
            return eips
        elif mode in ["rfr", "rfr-rfr", "rfr-gbm"]:
            mu, var = rfr_score_estimator.predict_mean_variance(points)
            mu = mu.reshape(-1)
            var = var.reshape(-1)
            sigma = np.sqrt(var)
            diff = mu - np.max(observed_Y)
            Z = diff / sigma
            ei = diff * norm.cdf(Z) + sigma * norm.pdf(Z)

            if mode == "rfr-rfr":
                run_time = rfr_time_estimator.predict(points).reshape(-1)
                eips = ei / np.maximum(0, run_time)
                eips[observed] = -1
                return eips

            elif mode == "rfr-gbm":
                run_time = lgbm_time_estimator.predict(points).reshape(-1)
                eips = ei / np.maximum(0, run_time)
                eips[observed] = -1
                return eips

            ei[observed] = -1
            return ei
        raise RuntimeError(f"Mode {mode} unknown.")


    results = {}
    fitting_times_per_task = {}
    prediction_times_per_task = {}
    evaluation_times_per_task = {}
    for task_id in tqdm(unique_groups):
        # Select by task
        indices = np.where(groups == task_id)[0]
        X_task = X[indices]
        Y_task = Y[indices]
        duration_task = np.array(duration)[indices]

        # Start with 3 random data points
        observed = np.zeros(len(X_task)).astype(bool)
        observed[:startup_rounds] = True
        observed_X = X_task[:startup_rounds].tolist()
        observed_Y = Y_task[:startup_rounds].tolist()
        observed_duration = duration_task[:startup_rounds].tolist()

        fitting_times = np.zeros(startup_rounds).tolist()
        prediction_times = np.zeros(startup_rounds).tolist()

        for i in range(1000):
            if i % 2 == 0 or no_random_sampling:
                start = time.time()
                if mode == "gbqr":
                    lgbm_score_estimator.fit(np.array(observed_X), np.array(observed_Y))
                elif mode == "gbqr-median":
                    lgbm_score_estimator.fit(np.array(observed_X), np.array(observed_Y))
                    lgbm_time_estimator.fit(np.array(observed_X), np.log(np.array(observed_duration) + 1))
                else:
                    rfr_score_estimator.fit(np.array(observed_X), np.array(observed_Y))

                    if mode == "rfr-gbm":
                        lgbm_time_estimator.fit(np.array(observed_X), np.log(np.array(observed_duration) + 1))
                    elif mode == "rfr-rfr":
                        rfr_time_estimator.fit(np.array(observed_X), np.log(np.array(observed_duration) + 1))
                fitting_times.append(time.time() - start)

                start = time.time()
                eis = predict_ei(X_task)
                prediction_times.append(time.time() - start)

                index = np.argmax(eis)
            else:
                index = np.random.choice(np.where(~observed)[0])

            observed[index] = True
            observed_X.append(X_task[index])
            observed_Y.append(Y_task[index])
            observed_duration.append(duration_task[index])
            # print(i, index, X_task[index].tolist(), Y_task[index])
            # print(index, end=",")

        results[task_id] = np.maximum.accumulate(np.array(observed_Y) / np.max(Y_task))
        fitting_times_per_task[task_id] = fitting_times
        prediction_times_per_task[task_id] = prediction_times
        evaluation_times_per_task[task_id] = observed_duration

    fit_frame = pd.DataFrame(fitting_times_per_task)
    pred_frame = pd.DataFrame(prediction_times_per_task)
    duration_frame = pd.DataFrame(evaluation_times_per_task)
    score_frame = pd.DataFrame(results)

    prefix = mode
    if not no_random_sampling:
        prefix += "-interleaved"
    fit_frame.to_csv(f"../simulation/1000-{prefix}-fit-time.csv")
    pred_frame.to_csv(f"../simulation/1000-{prefix}-time.csv")
    duration_frame.to_csv(f"../simulation/1000-{prefix}-eval-time.csv")
    score_frame.to_csv(f"../simulation/1000-{prefix}-scores.csv")
