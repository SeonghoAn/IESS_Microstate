import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statannotations.Annotator import Annotator
import pickle
import copy
import itertools
import scipy

col_list = ['etiology', 'Intractable epilepsy = 1', 'Profound delayed development = 1', 'IS_Free_ANY = 1',
            'BASED pre', 'BASED post', 'Sex(M:1/F:2)']

def is_dataload(w_data, info):
    feat = []
    first = True
    col = None
    label = []
    for subj, data in w_data.items():
        row = info.loc[info['Name']==subj][col_list]
        tp = data['transition_matrix']
        f  = copy.deepcopy(data)
        f.pop('unlabeled')
        f.pop('transition_matrix')
        tp_fl, fl_col = tp_flatten(tp['tm'], tp['cluster_name'])
        d = list(f.values())
        d.extend(tp_fl)
        d.extend(row.values[0].tolist())
        feat.append(d)
        if first:
            col = list(f.keys())
            col.extend(fl_col)
            col.extend(list(row.columns))
            first = False
    return pd.DataFrame(feat, columns=col)


def tp_flatten(tp, col):
    tp_fl = []
    fl_col = []
    for i in range(len(col)):
        for j in range(len(col)):
            if i == j:
                continue
            else:
                tp_fl.append(tp[i, j])
                fl_col.append('{}_to_{}'.format(col[i], col[j]))
    return tp_fl, fl_col

def nl_dataload(w_data):
    feat = []
    first = True
    col = None
    label = []
    for subj, data in w_data.items():
        tp = data['transition_matrix']
        f  = copy.deepcopy(data)
        f.pop('unlabeled')
        f.pop('transition_matrix')
        tp_fl, fl_col = tp_flatten(tp['tm'], tp['cluster_name'])
        d = list(f.values())
        d.extend(tp_fl)
        d.extend([-1 for i in range(len(col_list))])
        feat.append(d)
        if first:
            col = list(f.keys())
            col.extend(fl_col)
            col.extend(col_list)
            first = False
    return pd.DataFrame(feat, columns=col)

if __name__ == '__main__':
    with open('./feat_dict.pkl', 'rb') as f:
        data = pickle.load(f)
    info = pd.read_csv('./add_subj_info.csv') # Data is not available
    is_df = is_dataload(data['IS'], info)
    is_df['Infantile Spasms'] = [1 for i in range(is_df.shape[0])]
    nl_df = nl_dataload(data['Normal'])
    nl_df['Infantile Spasms'] = [0 for i in range(nl_df.shape[0])]
    save_df = pd.concat([is_df, nl_df], axis=0)
    save_df.to_csv('./feat.csv', encoding='utf-8')