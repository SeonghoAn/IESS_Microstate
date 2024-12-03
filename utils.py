from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import random
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import copy

col_list = ['etiology', 'Intractable epilepsy = 1', 'Profound delayed development = 1', 'IS_Free_ANY = 1',
            'BASED pre', 'BASED post', 'Infantile Spasms']

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_data(path, category):
    df = pd.read_csv(path, index_col=0)
    if category == 'etiology':
        df = df.loc[df[category] != -1]
        label = df[category].values
        idx = np.where(label == 2)
        label[idx] = 0
        # label = label + 1
    elif category == 'Infantile Spasms':
        label = df[category].values
    else:
        df = df.loc[df[category] != -1]
        label = df[category].values
        # label = label + 1
    df = df.drop(col_list, axis=1)
    return df, label


def kfold_cv_fs_mi(X, y, model, n, mi_score, k=5):
    X = copy.deepcopy(X)
    y = copy.deepcopy(y)
    mi_score = copy.deepcopy(mi_score)
    kf = StratifiedKFold(n_splits=k)
    kf.get_n_splits(X)
    y_pred = []
    y_true = []
    y_prob = []
    fs_col = X.columns
    fs_idx = np.argsort(mi_score)[-n:]
    for i, (train_idx, test_idx) in enumerate(tqdm(kf.split(X, y))):
        X_scaler = StandardScaler()
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y[train_idx]
        X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y[test_idx]
        X_train, X_test = X_train[fs_col[fs_idx]], X_test[fs_col[fs_idx]]
        X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns=fs_col[fs_idx])
        X_test = pd.DataFrame(X_scaler.transform(X_test), columns=fs_col[fs_idx])
        m = copy.deepcopy(model)
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        prob = m.predict_proba(X_test)
        y_pred.extend(pred)
        y_true.extend(y_test)
        y_prob.extend(prob)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def mibif_kfold(X, y, k=5):
    kf = StratifiedKFold(n_splits=k)
    kf.get_n_splits(X)
    mi_score = np.zeros(X.shape[1])
    for i, (train_idx, test_idx) in enumerate(tqdm(kf.split(X, y))):
        X_scaler = StandardScaler()
        X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y[train_idx]
        X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns=X_train.columns)
        _, fs_score = mibif(X_train, y_train, 1)
        mi_score += fs_score
    return mi_score / k

def conf_eval(conf):
    acc = (conf[0, 0] + conf[1, 1]) / conf.sum()
    sens = conf[0, 0] / (conf[0, 0] + conf[0, 1])
    spec = conf[1, 1] / (conf[1, 0] + conf[1, 1])
    return acc, sens, spec

def mibif(X_train, y_train, n):
    score = mutual_info_classif(X_train, y_train, discrete_features=False)
    idx = np.argsort(score)[-n:]
    return idx, score