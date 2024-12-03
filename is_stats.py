import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import copy
import itertools
import scipy
plt.rcParams['font.family'] = 'Times New Roman'
plt.rc('font', size=13, weight='bold')

col_list = ['etiology', 'Intractable epilepsy = 1', ]


def is_dataload(w_data, info):
    feat = []
    first = True
    col = None
    for subj, data in w_data.items():
        row = info.loc[info['Name'] == subj][col_list]
        temp = copy.deepcopy(data)
        temp.pop('unlabeled')
        temp.pop('transition_matrix')
        if first:
            col = list(temp.keys())
            col.extend(list(row.columns))
            first = False
        d = list(temp.values())
        d.extend(row.values[0].tolist())
        feat.append(d)
    return pd.DataFrame(feat, columns=col)


def nl_dataload(w_data):
    feat = []
    first = True
    col = None
    label = []
    for subj, data in w_data.items():
        f = copy.deepcopy(data)
        f.pop('unlabeled')
        f.pop('transition_matrix')
        d = list(f.values())
        d.extend([-1 for i in range(len(col_list))])
        feat.append(d)
        if first:
            col = list(f.keys())
            col.extend(col_list)
            first = False
    return pd.DataFrame(feat, columns=col)


def dataloader(w_data, info):
    is_df = is_dataload(w_data['IS'], info)
    nl_df = nl_dataload(w_data['Normal'])
    return pd.concat([nl_df, is_df], axis=0)


def load_per_group(df, group):
    c = list(df.columns)
    for r in col_list:
        c.remove(r)
    c.append(group)
    return df[c]


def flatten(data, label, l_name):
    f_data = []
    ms = []
    labels = []
    for i in range(data.shape[0]):
        temp = data.iloc[i]
        f_data.extend(temp.values)
        ms.extend([j.split('_')[0] for j in temp.index])
        labels.extend([label.iloc[i] for j in range(temp.shape[0])])
    return pd.DataFrame({'data': np.array(f_data), 'ms': np.array(ms), l_name: labels})


def normality_test(data, l_name):
    print('Category:', l_name)
    temp = data[l_name]
    data = data.drop(col_list, axis=1)
    col = list(data.columns)
    data[l_name] = temp
    if l_name == 'etiology':
        c_idx = np.where(data[l_name].values == 2)
        c_temp = data[l_name].values
        c_temp[c_idx] = 0
        data[l_name] = c_temp
    class_name = np.unique(data[l_name].values)
    arr = []  # shape: (class, subj, feature)
    for c in class_name:
        arr.append(data.loc[data[l_name] == c][col].values)
    for i, a in enumerate(arr):
        for j in range(len(col)):
            p = scipy.stats.shapiro(a[:, j]).pvalue
            nor = 'True' if p < 0.05 else 'False'
            if nor == 'True':
                print('Normality test | Class: {}, Feature: {}, p-value: {}, test: {}'.format(class_name[i], col[j], p,
                                                                                              nor))


def kruskal_test(data, l_name):
    print('Category:', l_name)
    temp = data[l_name]
    data = data.drop(col_list, axis=1)
    col = list(data.columns)
    data[l_name] = temp
    if l_name == 'etiology':
        c_idx = np.where(data[l_name].values == 2)
        c_temp = data[l_name].values
        c_temp[c_idx] = 0
        data[l_name] = c_temp
    class_name = np.unique(data[l_name].values)
    arr = []  # shape: (class, subj, feature)
    for c in class_name:
        arr.append(data.loc[data[l_name] == c][col].values)
    col_arr = []
    f_arr = []
    p_arr = []
    ast_arr = []
    for i in range(len(col)):
        f, p = scipy.stats.kruskal(arr[0][:, i], arr[1][:, i], arr[2][:, i])
        if p >= 0.05:
            ast = 'ns'
        elif p >= 0.01:
            ast = '*'
        elif p >= 0.001:
            ast = '**'
        elif p >= 0.0001:
            ast = '***'
        else:
            ast = '****'
        col_arr.append(col[i])
        f_arr.append(round(f, 4))
        p_arr.append(p)
        ast_arr.append(ast)
    save_df = pd.DataFrame()
    save_df['feature'] = col_arr
    save_df['F-value'] = f_arr
    save_df['p-value'] = p_arr
    save_df['asterisk'] = ast_arr
    save_df.to_csv('./is_stats/{}.csv'.format(l_name), encoding='utf-8')


def _stats_df(normal, IS, columns):
    pv_list = []
    fv_list = []
    c_list = []
    for c in columns:
        f, p = scipy.stats.mannwhitneyu(normal[c].values, IS[c].values)
        pv_list.append(p)
        fv_list.append(f)
        c_list.append(c)
    df = pd.DataFrame()
    df['feature'] = c_list
    df['u-stat'] = fv_list
    df['p-value'] = pv_list
    return df


def stats_df(df, l_name):
    map_dict = {'etiology': 'eti', 'Intractable epilepsy = 1': 'ie'}
    mean_corr = [i for i in df.columns if 'mean_corr' in i]
    gev = [i for i in df.columns if 'gev' in i]
    occur = [i for i in df.columns if 'occurrences' in i]
    timecov = [i for i in df.columns if 'timecov' in i]
    meandurs = [i for i in df.columns if 'meandurs' in i]
    if not map_dict[l_name] == 'sex':
        if l_name == 'etiology':
            c_idx = np.where(df[l_name].values == 2)
            c_temp = df[l_name].values
            c_temp[c_idx] = 0
            df[l_name] = c_temp
            label_dict = {-1: 'Normal', 0: 'Non struc-meta', 1: 'Struc-meta'}
        elif map_dict[l_name] == 'pdd':
            label_dict = {-1: 'Normal', 0: 'No PDD', 1: 'PDD'}
        sdf_list = []
        sname_list = []
        label = [(-1, 0), (0, 1), (-1, 1)]
        for f in [mean_corr, gev, occur, timecov, meandurs]:
            df_list = []
            pv_list = []
            for i, j in label:
                temp1 = df.loc[df[l_name] == i][f]
                temp2 = df.loc[df[l_name] == j][f]
                df_t = _stats_df(temp1, temp2, f)
                df_list.append(df_t)
                pv_list.extend(df_t['p-value'].values.reshape(-1).tolist())
            pv_list = scipy.stats.false_discovery_control(pv_list)
            for k in range(len(df_list)):
                df_list[k]['p-value'] = pv_list[k * 5:k * 5 + 5]
                ast_list = []
                for pv in pv_list[k * 5:k * 5 + 5]:
                    if pv < 0.0001:
                        ast_list.append('*' * 4)
                    elif pv < 0.001:
                        ast_list.append('*' * 3)
                    elif pv < 0.01:
                        ast_list.append('*' * 2)
                    elif pv < 0.05:
                        ast_list.append('*' * 1)
                    else:
                        ast_list.append('ns')
                df_list[k]['asterisk'] = ast_list
            sdf_list.append(df_list)
        for s in range(3):
            temp = [sdf_list[0][s], sdf_list[1][s], sdf_list[2][s], sdf_list[3][s], sdf_list[4][s]]
            sdf = pd.concat(temp, axis=0).reset_index(drop=True)
            sdf.to_csv(
                './is_stats/{}/{}_vs_{}.csv'.format(map_dict[l_name], label_dict[label[s][0]], label_dict[label[s][1]]))
    #             sname_list.append('./is_stats/{}/{}_vs_{}.csv'.format(map_dict[l_name], label_dict[i], label_dict[j]))
    else:
        label_dict = {1: 'Male', 2: 'Female'}
        df_list = []
        for f in [mean_corr, gev, occur, timecov, meandurs]:
            temp1 = df.loc[df[l_name] == 1][f]
            temp2 = df.loc[df[l_name] == 2][f]
            df_list.append(_stats_df(temp1, temp2, f))
        save_df = pd.concat(df_list, axis=0).reset_index(drop=True)
        save_df['p-value'] = scipy.stats.false_discovery_control(save_df['p-value'].values)
        save_df.to_csv('./is_stats/{}/{}_vs_{}.csv'.format(map_dict[l_name], label_dict[1], label_dict[2]),
                       encoding='utf-8')


def rename_label(df, labels, l_name):
    df = df.copy()
    lab = df[l_name].values.astype(str)
    for i in range(len(labels)):
        idx = np.where(lab == str(i - 1))
        lab[idx] = labels[i]
    df[l_name] = lab
    return df


def stats_plot(df, l_name):
    mean_corr = [i for i in df.columns if 'mean_corr' in i]
    gev = [i for i in df.columns if 'gev' in i]
    occur = [i for i in df.columns if 'occurrences' in i]
    timecov = [i for i in df.columns if 'timecov' in i]
    meandurs = [i for i in df.columns if 'meandurs' in i]

    mean_corr_df = flatten(df[mean_corr], df[l_name], l_name)
    gev_df = flatten(df[gev], df[l_name], l_name)
    occur_df = flatten(df[occur], df[l_name], l_name)
    timecov_df = flatten(df[timecov], df[l_name], l_name)
    meandurs_df = flatten(df[meandurs], df[l_name], l_name)

    df_list = [mean_corr_df, gev_df, occur_df, timecov_df, meandurs_df]
    feat_name = ['Mean Correlation', 'GEV (%)', 'Occurrence (Hz)', 'Time Coverage (%)', 'Mean Duration (sec)']
    map_dict = {'etiology': 'eti', 'Intractable epilepsy = 1': 'ie'}
    if l_name == 'etiology':
        c_idx = np.where(df[l_name].values == 2)
        c_temp = df[l_name].values
        c_temp[c_idx] = 0
        df[l_name] = c_temp
    if l_name == 'Sex(M:1/F:2)':
        df = df.loc[df[l_name] != -1]
    v = np.unique(df[l_name].values)
    ms_list = ['A', 'B', 'C', 'D', 'E']
    pairs = []
    for m in ms_list:
        comb = tuple(itertools.combinations(v, 2))
        t_comb = []
        for c in comb:
            temp = tuple([m for i in range(len(c))])
            t_comb.append(tuple(zip(temp, c)))
        pairs.extend(tuple(t_comb))
    labels = None
    if map_dict[l_name] == 'eti':
        labels = ['HC', 'Group NS', 'Group S']
    elif map_dict[l_name] == 'ie':
        labels = ['HC', 'Controlled epilepsy', 'Intractable epilepsy']

    for i, d in enumerate(df_list):
        plt.figure(facecolor='white')
        hue_order = labels
        sns.set_palette('gray_r')
        d = rename_label(d, labels, l_name)
        fontdict = {'fontweight': 'bold', 'fontsize': 12}
        ax = sns.boxplot(data=d, x='ms', y='data', hue=l_name, hue_order=hue_order, fliersize=2.0)
        plt.xlabel('Microstates', fontdict=fontdict)
        plt.ylabel(feat_name[i], fontdict=fontdict)
        plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
        plt.savefig('./is_stats/{}/{}.png'.format(map_dict[l_name], feat_name[i]), dpi=300, bbox_inches='tight')
        plt.close()
        print('Save plot : {} || {}'.format(l_name, feat_name[i]))


if __name__ == '__main__':
    with open('./feat_dict.pkl', 'rb') as f:
        data = pickle.load(f)
    info = pd.read_csv('./add_subj_info.csv') # Data is not available.
    df = dataloader(data, info)
    eti_df = load_per_group(df, 'etiology')
    ie_df = load_per_group(df, 'Intractable epilepsy = 1')
    kruskal_test(df, 'etiology')
    kruskal_test(df, 'Intractable epilepsy = 1')
    kruskal_test(df, 'Profound delayed development = 1')
    stats_plot(eti_df, 'etiology')
    stats_plot(ie_df, 'Intractable epilepsy = 1')