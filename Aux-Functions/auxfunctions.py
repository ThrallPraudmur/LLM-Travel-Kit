import pandas as pd
import numpy as np
import lightgbm
from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook


def train_lgb_optuna(X_train, y_train, X_valid, y_valid):
    train_df = lightgbm.Dataset(X_train, label=y_train)
    valid_df = lightgbm.Dataset(X_valid, label=y_valid)
    
    lgbm_fixed_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "random_state": 1,
        "n_jobs": 24,
    }
    
    def feval_gini(preds, train_data):
        result = 2*roc_auc_score(train_data.label, preds)-1
        return ("gini", result, True)
    
    fixed_sampler = optuna.samplers.TPESampler(seed=1, n_startup_trials=0)
    study = optuna.create_study(sampler=fixed_sampler, direction="minimize")
    model = lightgbm_optuna.train(
        lgbm_fixed_params, 
        train_set=train_df, 
        valid_sets=[valid_df],
        feval=feval_gini,
        callbacks=[early_stopping(25), log_evaluation(25)],
        study=study,
        verbosity=0
    )   
    return model

def train_lgb_raw(X_train, y_train, X_valid, y_valid, lgbm_params=None, feval=None):
    train_df = lightgbm.Dataset(X_train, label=y_train)
    valid_df = lightgbm.Dataset(X_valid, label=y_valid)
    
    default_lgbm_params = {
            "num_rounds": 500,
            "min_data_in_leaf": 1,
            "num_leaves": 24,
            "max_depth": 5,
            "reg_lambda": 10,
            "reg_alpha": 10,
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": 1,
            "n_jobs": 24,
            "verbosity": -1,
            "early_stopping_rounds": 10,
        }
    
    if lgbm_params is not None:
        for param_name, param_value in lgbm_params.items():
            default_lgbm_params[param_name] = param_value
    lgbm_params = default_lgbm_params
    
    
    def feval_gini(preds, train_data):
        result = 2*roc_auc_score(train_data.label, preds)-1
        return ("gini", result, True)
    
    def feval_roc_auc(preds, train_data):
        result = roc_auc_score(train_data.label, preds)
        return ("roc_auc", result, True)
    
    if feval is None:
        feval = feval_roc_auc
    
    model = lightgbm.train(
        lgbm_params, 
        train_set=train_df, 
        valid_sets=[valid_df],
        feval=feval,
        callbacks=[early_stopping(25), log_evaluation(25)],
    )   
    return model

def sample_diffirence_check(df_train, df_test, features=None, lgbm_params=None, feval=None):
    
    if features is None:
        features = list(df_train.columns)
        
    df_train['target'] = 0
    df_test['target'] = 1
    df = pd.concat([df_train, df_test]).reset_index(drop=True)
    train, valid_test = train_test_split(df, test_size=0.3, stratify=df.target)
    valid, test = train_test_split(valid_test, test_size=0.5, stratify=valid_test.target)
    
    print(pd.DataFrame({"sample": ["train", "valid", "test"], 
                        "n_obs": [len(train), len(valid), len(test)],
                        "target_rate": [train.target.mean(), valid.target.mean(), test.target.mean()],
                       }))
    print()
    
    discriminator = train_lgb_raw(train[features], train.target, 
                                  valid[features], valid.target, 
                                  lgbm_params=lgbm_params, feval=feval)
    
    print("train: ", roc_auc_score(train.target, discriminator.predict(train[features])))
    print("valid: ", roc_auc_score(valid.target, discriminator.predict(valid[features])))
    print("test : ", roc_auc_score(test.target, discriminator.predict(test[features])))
    
    return discriminator

def calc_weights(dates_series, min_weight):
    """
    Return exponential dacaying weights on time interval (min, max) of dates_series.
    Max weight: 1.0
    Min weight: min_weight (0.001 <= min_weight <= 1.0)
    """
    def __calc_weights(min_date, max_date, min_weight):
        assert 0.001 <= min_weight <= 1.0
        max_day = (max_date - min_date).days
        lambda_ = -(max_day-1)/np.log(min_weight)
        return pd.DataFrame({"weight": np.exp(-(max_day-np.linspace(1, max_day, max_day+1))/lambda_)}, 
                            index=range(0, max_day+1))

    def _calc_weights(dates_series, min_weight):
        min_date, max_date = dates_series.min(), dates_series.max()
        weights = __calc_weights(min_date, max_date, min_weight)
        dates = pd.DataFrame(pd.to_datetime(dates_series.apply(lambda x: f"{x.year}-{x.month:02d}-{x.day:02d}").drop_duplicates().sort_values())).reset_index(drop=True)
        return dates.join(weights)

    return pd.DataFrame(pd.to_datetime(dates_series.apply(lambda x: f"{x.year}-{x.month:02d}-{x.day:02d}"))
                       ).merge(_calc_weights(dates_series, min_weight), on='created_at', how='left').weight

def custom_gini(y_true, y_pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    roc_auc = roc_auc_score(y_true, y_pred) 
    return 2 * roc_auc - 1

def bnpl_stats_by_cat(df, catvar, *predvars, path="artefacts/"):
    """
    Returns pd.DataFrame  with GINI and hit rate of predvars by catvar.
    If path is not None then table will be saved in the dir.
    """
    
    stats = df.groupby(catvar).agg(cnt=("cart_id_bnpl", "count"),
                                   npl30_mean=("npl30", "mean"),
                                   price_mean=('cart_price', 'mean')
    )
    
    for predvar in predvars:
        ginis = df.groupby(catvar).apply(lambda x: custom_gini(x['npl30'], x[predvar].fillna(x[predvar].median())))
        ginis.name = f'GINI {predvar}'
        stats = stats.join(ginis)
        
    for predvar in predvars:
        hitrate = df.groupby(catvar).apply(lambda x: x[predvar].notnull().mean())
        hitrate.name = f'HR {predvar}'
        stats = stats.join(hitrate)

    final_table = stats.sort_values('cnt', ascending=False)
    if path is not None:
        final_table.to_csv(f"{path}/bnpl_stats_by_{catvar}.csv", index=True)
    
    return final_table

def gini_bootstrapped(model, X: pd.DataFrame, y, n_iter=1000) -> list:
    """
    Arguments:
        model - any model with sci-kit learn .predict_proba method
        X - pandas dataframe
        y - pandas series
        n_iter - number of measurements (num ginis)
    """
    ginis = []
    for j in range(n_iter):
        X_tmp = X.sample(frac=1.0, replace=True, random_state=j)
        y_tmp = y[X_tmp.index]
        if isinstance(model, lightgbm.LGBMClassifier):
            preds = model.predict_proba(X_tmp)[:, 1]
        elif isinstance(model, lightgbm.Booster):
            preds = model.predict(X_tmp)
        ginis.append(2*roc_auc_score(y_tmp, preds)-1)
    return ginis

def gini_bootstrapped_plot(ginis):
    """
    Function calcs confidence interval for bootstrapped ginis
    """
    gini_median = np.percentile(ginis, q=50)
    gini_p05 = np.percentile(ginis, q=5)
    gini_p95 = np.percentile(ginis, q=95)
    
    gini_desc_str = f"{gini_median:.2f} ({gini_p05:.2f}, {gini_p95:.2f})"
    
    plt.hist(ginis)
    plt.title(f"gini = {gini_desc_str}")
    plt.xlabel("GINI")
    plt.grid(True)
    plt.show()
    return gini_desc_str

def find_time_split(date_series, test_size=0.3, rtol=0.001):
    """
    Arguments:
        date_series - pandas series represents observation dates
        test_size - fraction of record for OOT sample
        rtol - relative tolerance of sample fractions
    Returns:
        Timestamp which splits the sample to two samples with fraction (1-test_size) ans test_size 
    """
    ub = date_series.max()
    lb = date_series.min()
    max_test_size = test_size*(1+rtol)
    min_test_size = test_size*(1-rtol)
    cur_test_size = 1.0
    while (cur_test_size > max_test_size) or (cur_test_size < min_test_size):
        
        median = lb + (ub-lb)/2
        
        if (median==lb) or (median==ub):
            break
        
        cur_test_size = (date_series>median).mean()
        
        if cur_test_size > max_test_size:
            lb = median
        elif cur_test_size < min_test_size:
            ub = median
            
    return median

def plot_param_in_time(data: pd.DataFrame, uid: str, timevar: str, freq: str="1M", figsize=(12,5), **func2param: dict) -> pd.DataFrame:
    """
    Returns aggregates by datevar interval.
    data: pd.DataFrame / data for analysis
    uid: str / identificator for counts
    timevar: str / datetime variable
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    func2param: dict / mapping for aggregate functions and data variables
    """
    stat_params = data.groupby(pd.Grouper(key='created_at', freq=freq))\
    .apply(lambda x: pd.Series({col : func(x.loc[:, col]) for col, func in func2param.items()}))
            
    fig, ax1 = plt.subplots(figsize=figsize)
    time_values = range(len(stat_params.index))
    time_labels = stat_params.index.date

    # Draw counts
    print(stat_params.columns)
    bar_temp_ = ax1.bar(time_values, stat_params[uid], alpha=0.4)
    xtickx_temp_ = ax1.set_xticks(time_values, labels=time_labels, rotation=90)
    ax1.set_ylabel(f"Counts (id)")
    plt.grid(True)

    # Draw param values
    ax2 = ax1.twinx()
    params_0_1 = [c for c in stat_params.columns if c != uid]
    for c in params_0_1:
        ax2.plot(time_values, stat_params[c], '-o', label=f"{c}")
    ax2.set_ylabel(f"Param Values")

    ax2.legend(loc=(0.85,0.85))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stat_params

def plot_gini_in_time(data: pd.DataFrame, uid: str, timevar: str, targetvar: str, predvars: list, freq: str="1M", figsize=(12,5)) -> pd.DataFrame:
    """
    Returns aggregates by datevar interval.
    data: pd.DataFrame / data for analysis
    uid: str / identificator for counts
    timevar: str / datetime variable
    freq : str / frequency object, defaults to None
        This will groupby the specified frequency if the target selection
        (via key or level) is a datetime-like object. For full specification
        of available frequencies, please see `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    func2param: dict / mapping for aggregate functions and data variables
    """
    def gini(y_true, y_pred):
        if len(y_pred) == 0 or np.unique(y_true).shape[0] < 2:
            return np.nan
        roc_auc = roc_auc_score(y_true, y_pred) 
        return 2 * roc_auc - 1
    
    stat_params = data.groupby(pd.Grouper(key='created_at', freq=freq))\
    .apply(lambda x: pd.Series({**{uid: len(x), 
                                targetvar: 100*x.loc[:, targetvar].mean()}, 
                                **{predvar: gini(x.loc[:, targetvar], x.loc[:, predvar].fillna(0.0))
                                   for predvar in predvars}
                               }))
    # 2*roc_auc_score(x.loc[:, targetvar], x.loc[:, predvar])-1 
    fig, ax1 = plt.subplots(figsize=figsize)
    time_values = range(len(stat_params.index))
    time_labels = stat_params.index.date

    # Draw counts
    print(stat_params.columns)
    bar_temp_ = ax1.bar(time_values, stat_params[uid], alpha=0.4)
    xtickx_temp_ = ax1.set_xticks(time_values, labels=time_labels, rotation=90)
    ax1.set_ylabel(f"Counts (id)")
    
    def add_labels(x, y):
        for i in range(len(x)):
            plt.text(i, y[i], f"{stat_params[targetvar][i]:.1f}%", ha='right')
    add_labels(time_values, stat_params[uid])
    
    plt.grid(True)

    # Draw param values
    ax2 = ax1.twinx()
    for predvar in predvars:
        ax2.plot(time_values, stat_params[predvar], '--o', label=f"{predvar}")
    ax2.set_ylabel(f"GINI")
    ax2.grid(True)
    ax2.legend(loc=(0.85,0.85))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stat_params

def describe_samples(uid: str, datevar: str, target: str, **samples: dict) -> pd.DataFrame:
    """
    Function returns pd.DataFrame with statistic by sample.
    samples arg is a dict with sample name, sample pd.DataFrame
    """
    n_obs = [len(sample) for sample in samples.values()]
    total_n_obs = np.sum(n_obs)
    n_unique_clients = [sample[uid].nunique() for sample in samples.values()]
    total_unique_clients = np.sum(n_unique_clients)
    
    description = {"n obs": n_obs,
                   "% obs": [100*n/total_n_obs for n in n_obs],
                   "n unique clients": n_unique_clients,
                   "% unique clients": [100*n/total_unique_clients for n in n_unique_clients],
                   "n pos obs (target)": [sample[target].sum() for sample in samples.values()],
                   "Target Rate": [sample[target].mean() for sample in samples.values()],
                   "Target Date MIN": [sample[datevar].min() for sample in samples.values()],
                   "Target Date MAX": [sample[datevar].max() for sample in samples.values()],
                  }
    return pd.DataFrame(description, index=[name for name, _ in samples.items()]).T

def plot_reliability_curve(y_true, y_pred, n_bins, strategy):
    """
    Plots and saves reliability plot with predictions/true labels histograms.
    """
    
    assert strategy in ('uniform', 'quantile')

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy=strategy)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(prob_true, prob_pred, "-o")
    axs[0].plot([0, 1], [0, 1], "--")
    axs[0].set_xlim(0.0, 1.0)
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_title(f"Reliability Curve (n_bins={n_bins})")
    axs[0].set_xlabel("avg TRUE")
    axs[0].set_ylabel("avg PRED")
    axs[0].grid(True)

    axs[1].hist(y_pred, alpha=0.5, bins=10, label="Estimation")
    axs[1].hist(y_true, alpha=0.5, bins=10, label=f"Target, mean={np.mean(y_true):.2f}")
    axs[1].set_xlabel("Scores")
    axs[1].grid(True)
    plt.legend()

    fig.tight_layout()

    fig.savefig("reliability_curve.png")
    
class CustomRFE():
    
    def __init__(self, estimator, step=1, min_features=1, threshold=0.05, scoring_function=None, cv=None):
        assert (min_features >= 1)
        self.estimator = estimator
        self.step = step
        self.min_features = min_features
        self.threshold = threshold
        self.scoring_function = self._scoring_function(scoring_function)
        self.stats = []
        self.is_fitted = None
        
    def _scoring_function(self, scoring_function):
        if scoring_function:
            return scoring_function
        
        def gini(y_true, y_pred):
            return 2*roc_auc_score(y_true, y_pred)-1
        
        return gini
    
    def _scoring(self, X, y):
        return self.scoring_function(y, self.estimator.predict_proba(X)[:,1])
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        columns = X.columns
        mask = [True]*len(columns)
        while sum(mask)>=self.min_features:
            X_tmp = X.loc[:, mask]
            self.estimator.fit(X_tmp, y)

            score = self._scoring(X_tmp, y)
            
            self.stats.append((sum(mask), columns[mask], score))
            
            worst_features = [feat for feat, _ in sorted(zip(self.estimator.feature_name_, self.estimator.feature_importances_),
                                     key=lambda x: x[1],
                                     reverse=True
                                    )[-self.step:]]
            
            mask = [False if feat in worst_features else b for b, feat in zip(mask, columns)]
            
        self.is_fitted = True
        self._auto_feature_selection()
        return self
    
    def _auto_feature_selection(self):
        scores = [el[2] for el in self.get_stats()]
        diffs = [(scores[j]-scores[j+1]) / (scores[0]) for j in range(len(scores)-1)]
        self.best_stats_idx = (np.cumsum(diffs)<self.threshold).sum()
        self.best_feature_subspace = self.get_stats()[self.best_stats_idx][1]
        
    def get_best_feature_subspace(self):
        return self.best_feature_subspace
    
    def plot_score_versus_features(self):
        n_features = len(self.stats)
        scores = [el[2] for el in self.stats]
        ymin = np.min(scores)
        ymax = np.max(scores)
        
        if not self.is_fitted:
            raise Exception("CustomRFE has not been fitted yet.")
        
        # plot auto feature selection line
        plt.vlines(n_features-(n_features-self.best_stats_idx), 
                   ymin, ymax, colors='r', linestyles='dashed')
        plt.text(self.best_stats_idx, 
                 self.stats[self.best_stats_idx][2], 
                 f"auto selection,\nscore drop <={self.threshold}", ha='left')
        
        plt.plot(range(n_features), scores)
        plt.xticks(ticks=range(n_features), labels=[el[0] for el in self.stats])
        plt.grid(True)
        plt.xlabel("N features")
        plt.ylabel("Score")
        plt.title(f"Score vs N features")
    
    def get_stats(self):
        if not self.is_fitted:
            raise Exception("CustomRFE has not been fitted yet.")
        return self.stats
    
class CustomRFEValid():
    
    def __init__(self, estimator, step=1, min_features=1, threshold=0.05, scoring_function=None):
        assert (min_features >= 1)
        self.estimator = estimator
        self.step = step
        self.min_features = min_features
        self.threshold = threshold
        self.scoring_function = self._scoring_function(scoring_function)
        self.stats = []
        self.is_fitted = None
        
    def _scoring_function(self, scoring_function):
        if scoring_function:
            return scoring_function
        
        def gini(y_true, y_pred):
            return 2*roc_auc_score(y_true, y_pred)-1
        
        return gini
    
    def _scoring(self, X, y, X_valid, y_valid):
        train_score = self.scoring_function(y, self.estimator.predict_proba(X)[:,1])
        valid_score = self.scoring_function(y_valid, self.estimator.predict_proba(X_valid)[:,1])
        return train_score, valid_score
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        columns = X.columns
        mask = [True]*len(columns)
        while sum(mask)>=self.min_features:
            X_tmp = X.loc[:, mask]
            self.estimator.fit(X_tmp, y)

            train_score, valid_score = self._scoring(X_tmp, y, X_valid.loc[:, mask], y_valid)
            
            self.stats.append((sum(mask), columns[mask], valid_score, train_score))
            
            worst_features = [feat for feat, _ in sorted(zip(self.estimator.feature_name_, self.estimator.feature_importances_),
                                     key=lambda x: x[1],
                                     reverse=True
                                    )[-self.step:]]
            
            mask = [False if feat in worst_features else b for b, feat in zip(mask, columns)]
            
        self.is_fitted = True
        self._auto_feature_selection()
        return self
    
    def _auto_feature_selection(self):
        scores = [el[2] for el in self.get_stats()]
        diffs = [(scores[0]-scores[j]) / (scores[0]) for j in range(len(scores)-1)]
        self.best_stats_idx = 0
        for j, diff in enumerate(diffs):
            if diff<=self.threshold:
                self.best_stats_idx=j
        self.best_feature_subspace = self.get_stats()[self.best_stats_idx][1]
        
    def get_best_feature_subspace(self):
        return self.best_feature_subspace
    
    def plot_score_versus_features(self):
        n_features = len(self.stats)
        valid_scores = [el[2] for el in self.stats]
        train_scores = [el[3] for el in self.stats]
        ymin = np.min(valid_scores)
        ymax = np.max(valid_scores)
        
        if not self.is_fitted:
            raise Exception("CustomRFE has not been fitted yet.")
            
        fig = plt.figure(figsize=(12, 6))
        
        # plot auto feature selection line
        plt.vlines(n_features-(n_features-self.best_stats_idx), 
                   ymin, ymax, colors='r', linestyles='dashed')
        plt.text(self.best_stats_idx, 
                 self.stats[self.best_stats_idx][2], 
                 f"auto selection,\nscore drop <={self.threshold}", ha='left')
        
        plt.plot(range(n_features), valid_scores, label='valid')
        plt.plot(range(n_features), train_scores, label='train')
        plt.xticks(ticks=range(n_features), labels=[el[0] for el in self.stats], rotation=45)
        plt.grid(True)
        plt.xlabel("N features")
        plt.ylabel("Score")
        plt.title(f"Score vs N features")
        plt.legend(loc='lower left')
    
    def get_stats(self):
        if not self.is_fitted:
            raise Exception("CustomRFE has not been fitted yet.")
        return self.stats
    
class CustomRFECV():
    
    def __init__(self, estimator, step=1, min_features=1, threshold=0.05, scoring_function=None, cv=None):
        assert (min_features >= 1)
        self.estimator = estimator
        self.step = step
        self.min_features = min_features
        self.threshold = threshold
        self.scoring_function = self._scoring_function(scoring_function)
        self.cv = self._cv_mock(cv)
        self.stats = []
        self.is_fitted = None
        
    def _scoring_function(self, scoring_function):
        if scoring_function:
            return scoring_function
        
        def gini(y_true, y_pred):
            return 2*roc_auc_score(y_true, y_pred)-1
        
        return gini
    
    def _cv_mock(self, cv):
        if cv:
            return cv
        
        cv = KFold(n_splits=5, shuffle=True, random_state=1)
        
        return cv
    
    def _scoring(self, X, y):
        cv_scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring='roc_auc')
        cv_scores = [2*score-1 for score in cv_scores]
        score_mean = np.mean(cv_scores)
        score_std = np.std(cv_scores)
        return score_mean, score_std
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        columns = X.columns
        mask = [True]*len(columns)
        while sum(mask)>=self.min_features:
            X_tmp = X.loc[:, mask]
            self.estimator.fit(X_tmp, y)

            score, score_std = self._scoring(X_tmp, y)
            
            self.stats.append((sum(mask), columns[mask], score, score_std))
            
            worst_features = [feat for feat, _ in sorted(zip(self.estimator.feature_name_, self.estimator.feature_importances_),
                                     key=lambda x: x[1],
                                     reverse=True
                                    )[-self.step:]]
            
            mask = [False if feat in worst_features else b for b, feat in zip(mask, columns)]
            
        self.is_fitted = True
        self._auto_feature_selection()
        return self
    
    def _auto_feature_selection(self):
        scores = [el[2] for el in self.get_stats()]
        diffs = [(scores[j]-scores[j+1]) / (scores[0]) for j in range(len(scores)-1)]
        self.best_stats_idx = (np.cumsum(diffs)<self.threshold).sum()
        self.best_feature_subspace = self.get_stats()[self.best_stats_idx][1]
        
    def get_best_feature_subspace(self):
        return self.best_feature_subspace
    
    def plot_score_versus_features(self):
        n_features = len(self.stats)
        scores = np.array([el[2] for el in self.stats])
        stds = np.array([el[3] for el in self.stats])
        ymin = np.min(scores)
        ymax = np.max(scores)
        
        if not self.is_fitted:
            raise Exception("CustomRFE has not been fitted yet.")
        
        # plot auto feature selection line
        plt.vlines(n_features-(n_features-self.best_stats_idx), 
                   ymin, ymax, colors='r', linestyles='dashed')
        plt.text(self.best_stats_idx, 
                 self.stats[self.best_stats_idx][2], 
                 f"auto selection,\nscore drop <={self.threshold}", ha='left')
        
        plt.plot(range(n_features), scores, label='mean score')
        plt.plot(range(n_features), scores-stds, label='lower bound')
        plt.plot(range(n_features), scores+stds, label='upper bound')
        plt.xticks(ticks=range(n_features), labels=[el[0] for el in self.stats])
        plt.grid(True)
        plt.xlabel("N features")
        plt.ylabel("Score")
        plt.title(f"Score vs N features")
        plt.legend(loc='lower left')
    
    def get_stats(self):
        if not self.is_fitted:
            raise Exception("CustomRFE has not been fitted yet.")
        return self.stats
    
class CustomFFSValid():
    
    def __init__(self, estimator, step=1, max_features=15, threshold=0.05, scoring_function=None):
        self.estimator = estimator
        self.step = step
        self.max_features = max_features
        self.threshold = threshold
        self.scoring_function = self._scoring_function(scoring_function)
        self.stats = []
        self.is_fitted = None
        
    def _scoring_function(self, scoring_function):
        if scoring_function:
            return scoring_function
        
        def gini(y_true, y_pred):
            return 2*roc_auc_score(y_true, y_pred)-1
        
        return gini
    
    def _scoring(self, X, y, X_valid, y_valid):
        train_score = self.scoring_function(y, self.estimator.predict_proba(X)[:,1])
        valid_score = self.scoring_function(y_valid, self.estimator.predict_proba(X_valid)[:,1])
        return train_score, valid_score
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        columns = X.columns
        selected_features = []
        for i in tqdm_notebook(range(self.max_features)):
            
            best_feature = None
            best_valid_score = 0.0
            best_train_score = 0.0
            for feat in columns:
                
                if feat in selected_features:
                    continue
            
                X_tmp = X.loc[:, selected_features+[feat]]
                self.estimator.fit(X_tmp, y)

                train_score, valid_score = self._scoring(X_tmp, y, X_valid.loc[:, selected_features+[feat]], y_valid)
                
                if best_valid_score < valid_score:
                    best_feature = feat
                    best_valid_score = valid_score
                    best_train_score = train_score
            
            if best_feature:
                selected_features = selected_features+[best_feature]
                self.stats.append((len(selected_features), 
                                   selected_features, 
                                   best_valid_score, best_train_score)
                                 )
            else:
                break
            
        self.is_fitted = True
        self._auto_feature_selection()
        return self
    
    def _auto_feature_selection(self):
        scores = [el[2] for el in self.get_stats()]
        diffs = [(scores[-1]-scores[j]) / (scores[-1]) for j in range(len(scores))]
        self.best_stats_idx = 0
        for j, diff in enumerate(diffs):
            if diff<=self.threshold:
                self.best_stats_idx=j
                break
        self.best_feature_subspace = self.get_stats()[self.best_stats_idx][1]
        
    def get_best_feature_subspace(self):
        return self.best_feature_subspace
    
    def plot_score_versus_features(self):
        n_features = len(self.stats)
        valid_scores = [el[2] for el in self.stats]
        train_scores = [el[3] for el in self.stats]
        ymin = np.min(valid_scores)
        ymax = np.max(valid_scores)
        
        if not self.is_fitted:
            raise Exception("CustomFFS has not been fitted yet.")
            
        fig = plt.figure(figsize=(12, 6))
        
        # plot auto feature selection line
        plt.vlines(n_features-(n_features-self.best_stats_idx), 
                   ymin, ymax, colors='r', linestyles='dashed')
        plt.text(self.best_stats_idx, 
                 self.stats[self.best_stats_idx][2], 
                 f"auto selection,\nscore drop <={self.threshold}", ha='left')
        
        plt.plot(range(n_features), valid_scores, label='valid')
        plt.plot(range(n_features), train_scores, label='train')
        plt.xticks(ticks=range(n_features), labels=[el[0] for el in self.stats], rotation=45)
        plt.grid(True)
        plt.xlabel("N features")
        plt.ylabel("Score")
        plt.title(f"Score vs N features")
        plt.legend(loc='lower left')
    
    def get_stats(self):
        if not self.is_fitted:
            raise Exception("CustomFFS has not been fitted yet.")
        return self.stats
    
    
class CustomFFSValidLGBAPI():
    
    def __init__(self, step=1, max_features=15, scoring_function=None):
        self.step = step
        self.max_features = max_features
        self.scoring_function = self._scoring_function(scoring_function)
        self.stats = []
        self.is_fitted = None
        
    def _scoring_function(self, scoring_function):
        if scoring_function:
            return scoring_function
        
        def gini(y_true, y_pred):
            return 2*roc_auc_score(y_true, y_pred)-1
        
        return gini
    
    def _scoring(self, estimator, X, y, X_valid, y_valid):
        train_score = self.scoring_function(y, estimator.predict(X))
        valid_score = self.scoring_function(y_valid, estimator.predict(X_valid))
        return train_score, valid_score
    
    def _train_lgb_raw(self, X_train, y_train, X_valid, y_valid):
        train_df = lightgbm.Dataset(X_train, label=y_train)
        valid_df = lightgbm.Dataset(X_valid, label=y_valid)

        lgbm_fixed_params = {
            "num_rounds": 500, 
            "num_leaves": 24,
            "max_depth": 5,
            "reg_lambda": 30,
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": 1,
            "n_jobs": 24,
            "verbosity": -1,
        }

        def feval_gini(preds, train_data):
            result = 2*roc_auc_score(train_data.label, preds)-1
            return ("gini", result, True)

        model = lightgbm.train(
            lgbm_fixed_params, 
            train_set=train_df, 
            valid_sets=[valid_df],
            feval=feval_gini,
            callbacks=[early_stopping(25), log_evaluation(25)]
        )   
        return model
    
    def fit(self, X, y, X_valid=None, y_valid=None):
        columns = X.columns
        selected_features = []
        for i in tqdm_notebook(range(self.max_features)):
            
            best_feature = None
            best_valid_score = 0.0
            best_train_score = 0.0
            for feat in columns:
                
                if feat in selected_features:
                    continue
            
                X_tmp = X.loc[:, selected_features+[feat]]
                estimator = self._train_lgb_raw(X_tmp, y, X_valid.loc[:, selected_features+[feat]], y_valid)
                #self.estimator.fit(X_tmp, y)

                train_score, valid_score = self._scoring(estimator, X_tmp, y, X_valid.loc[:, selected_features+[feat]], y_valid)
                
                if best_valid_score < valid_score:
                    best_feature = feat
                    best_valid_score = valid_score
                    best_train_score = train_score
            
            if best_feature:
                selected_features = selected_features+[best_feature]
                self.stats.append((len(selected_features), 
                                   selected_features, 
                                   best_valid_score, best_train_score)
                                 )
            else:
                break
            
        self.is_fitted = True
        return self
    
    def _auto_feature_selection(self, threshold):
        self.threshold = threshold
        scores = [el[2] for el in self.get_stats()]
        diffs = [(scores[-1]-scores[j]) / (scores[-1]) for j in range(len(scores))]
        self.best_stats_idx = 0
        for j, diff in enumerate(diffs):
            if diff<=self.threshold:
                self.best_stats_idx=j
                break
        self.best_feature_subspace = self.get_stats()[self.best_stats_idx][1]
        
    def get_best_feature_subspace(self, threshold):
        if isinstance(threshold, float):
            self._auto_feature_selection(threshold)
        elif isinstance(threshold, int):
            self.threshold = None
            self.best_stats_idx = threshold-1
            self.best_feature_subspace = self.get_stats()[self.best_stats_idx][1]
        return self.best_feature_subspace
    
    def plot_score_versus_features(self):
        n_features = len(self.stats)
        valid_scores = [el[2] for el in self.stats]
        train_scores = [el[3] for el in self.stats]
        ymin = np.min(valid_scores)
        ymax = np.max(valid_scores)
        
        if not self.is_fitted:
            raise Exception("CustomFFS has not been fitted yet.")
            
        fig = plt.figure(figsize=(12, 6))
        
        # plot auto feature selection line
        plt.vlines(n_features-(n_features-self.best_stats_idx), 
                   ymin, ymax, colors='r', linestyles='dashed')
        plt.text(self.best_stats_idx, 
                 self.stats[self.best_stats_idx][2], 
                 f"auto selection,\nscore drop <={self.threshold}", ha='left')
        
        plt.plot(range(n_features), valid_scores, label='valid')
        plt.plot(range(n_features), train_scores, label='train')
        plt.xticks(ticks=range(n_features), labels=[el[0] for el in self.stats], rotation=45)
        plt.grid(True)
        plt.xlabel("N features")
        plt.ylabel("Score")
        plt.title(f"Score vs N features")
        plt.legend(loc='lower left')
    
    def get_stats(self):
        if not self.is_fitted:
            raise Exception("CustomFFS has not been fitted yet.")
        return self.stats