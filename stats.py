import numpy as np
import pandas as pd
import scipy
from statsmodels.sandbox.stats.multicomp import MultiComparison
from tqdm import tqdm

col_list = ['etiology', 'Infantile Spasms', 'Intractable epilepsy = 1']
rm_list = ['aaa'] # Data is not available.
eti_map = {-1: 'Normal', 0: 'Non-struc-meta', 1: 'Struc-meta'}
is_map = {0: 'Normal', 1: 'Infantile Spasms'}
ie_map = {-1: 'Normal', 0: 'Controlled epilepsy', 1: 'Interactable epilepsy'}


def read_data():
    path = './feat.csv'
    df = pd.read_csv(path, encoding='utf-8', index_col=0)
    groups = df[col_list]
    groups['etiology'].loc[(groups['etiology'] == 2)] = 0
    for c in df.columns:
        if 'mean_corr' in c:
            rm_list.append(c)
    feats = df.drop(rm_list, axis=1)
    return feats, groups


def save_stats():
    feat, group = read_data()
    feat_col = feat.columns
    group_col = group.columns
    for g in group_col:
        print(g)
        stats_df = None
        is_first = True
        if g == 'etiology':
            mapping = eti_map
        elif g == 'Infantile Spasms':
            mapping = is_map
        elif g == 'Intractable epilepsy = 1':
            mapping = ie_map
        for f in tqdm(feat_col):
            comp = MultiComparison(feat[f], group[g])
            result = comp.allpairtest(scipy.stats.mannwhitneyu, method='bonf')
            csv = result[0].as_csv()
            col = [c.strip() for c in csv.split('\n')[3].split(',')]
            row = [[c.strip() for c in row.split(',')] for row in csv.split('\n')[4:]]
            stats = pd.DataFrame(row, columns=col)
            stats['Feature'] = f
            if is_first:
                stats_df = stats
                is_first = False
            else:
                stats_df = pd.concat([stats_df, stats], axis=0)
        if g == 'Infantile Spasms':
            stats_df = rename_group_is(stats_df, mapping)
        else:
            stats_df = rename_group(stats_df, mapping)
        stats_df.to_csv('./stats/{}_stats_test.csv'.format(g), encoding='utf-8', index=False)


def rename_group(df, mapping):
    g1 = df['group1'].values
    g2 = df['group2'].values
    g1s = ['-1.0', '0.0']
    g2s = ['0.0', '1.0']
    for g in g1s:
        print(g, g1)
        g1_idx = np.where(g1 == g)
        g1[g1_idx] = mapping[int(float(g))]
    for g in g2s:
        g2_idx = np.where(g2 == g)
        g2[g2_idx] = mapping[int(float(g))]
    df['group1'] = g1
    df['group2'] = g2
    return df


def rename_group_is(df, mapping):
    g1 = df['group1'].values
    g2 = df['group2'].values
    g1_idx = np.where(g1 == '0.0')
    g1[g1_idx] = mapping[int(float('0.0'))]
    g2_idx = np.where(g2 == '1.0')
    g2[g2_idx] = mapping[int(float('1.0'))]
    df['group1'] = g1
    df['group2'] = g2
    return df


def save_mean():
    feat, group = read_data()
    feat_col = feat.columns
    group_col = group.columns
    for g in group_col:
        print(g)
        labels = np.unique(group[g].values)
        df = pd.concat([feat, group[g]], axis=1)
        save_df = pd.DataFrame()
        if g == 'etiology':
            mapping = eti_map
        elif g == 'Infantile Spasms':
            mapping = is_map
        for l in labels:
            val, name = [], []
            for f in tqdm(feat_col):
                temp = df.loc[df[g] == l][f].values
                val.append(scipy.stats.iqr(temp))
            save_df[mapping[l]] = val
        save_df['Feature'] = feat_col
        save_df.to_csv('./stats/{}_iqr.csv'.format(g), encoding='utf-8', index=False)



if __name__ == '__main__':
    save_mean()
    save_stats()