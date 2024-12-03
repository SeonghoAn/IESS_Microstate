import numpy as np
import pandas as pd
import scipy
from statsmodels.sandbox.stats.multicomp import MultiComparison
from tqdm import tqdm

col_list = ['etiology', ]
rm_list = ['aaa'] # Data is not available.

def read_data():
    path = './feat.csv'
    df = pd.read_csv(path, encoding='utf-8', index_col=0)
    groups = df[col_list]
    groups['etiology'].loc[(groups['etiology']==2)] = 0
    feats = df.drop(rm_list, axis=1)
    return feats, groups

def save_stats():
    feat, group = read_data()
    feat_col = feat.columns
    tp_col = []
    for c in feat_col:
        if '_to_' in c:
            tp_col.append(c)
    group_col = group.columns
    for g in group_col:
        print(g)
        stats_df = None
        is_first = True
        for f in tqdm(tp_col):
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
        stats_df.to_csv('./stats/{}_tp.csv'.format(g), encoding='utf-8', index=False)

def tp_fdr():
    groups = [(-1, 0), (-1, 1), (0, 1)]
    for cate in col_list:
        fdr_list = []
        df = pd.read_csv('./stats/{}_tp.csv'.format(cate), encoding='utf-8')
        for (i, j) in groups:
            pvs = df.loc[(df['group1']==i) & (df['group2']==j)]['pval_corr'].values
            fdr_list.append(scipy.stats.false_discovery_control(pvs))
        merge_list = []
        for k in range(len(fdr_list[0])):
            merge_list.append(fdr_list[0][k])
            merge_list.append(fdr_list[1][k])
            merge_list.append(fdr_list[2][k])
        df['pval_corr_fdr'] = np.array(merge_list)
        df.to_csv('./stats/{}_tp_fdr.csv'.format(cate), encoding='utf-8')

if __name__ == '__main__':
    save_stats()
    tp_fdr()