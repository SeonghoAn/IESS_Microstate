import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import copy
import scipy

plt.rcParams['font.family'] = 'Times New Roman'
plt.rc('font', size=12, weight='bold')


def nor_vs_is(w_data):
    feat = []
    col = None
    first = True
    for group, subj in w_data.items():
        for name, data in subj.items():
            temp = copy.deepcopy(data)
            temp.pop('unlabeled')
            temp.pop('transition_matrix')
            if first:
                col = list(temp.keys())
                first = False
            d = list(temp.values())
            d.append(group)
            feat.append(d)
    col.append('Group')
    return pd.DataFrame(feat, columns=col)


def nor_vs_is_tp(w_data):
    feat = []
    col = None
    first = True
    label = []
    for group, subj in w_data.items():
        for name, data in subj.items():
            temp = data['transition_matrix']
            if first:
                col = temp['cluster_name']
                first = False
            d = list(temp['tm'])
            feat.append(d)
            label.append(group)
    return np.array(feat), np.array(label), col


def tpm_stats(data, label, col):
    nor_idx = np.where(label == 'Normal')
    nor_data = data[nor_idx]
    is_idx = np.where(label == 'IS')
    is_data = data[is_idx]

    plt.figure(facecolor='white')
    ax = sns.heatmap(nor_data.mean(axis=0), annot=True, cmap="Blues", xticklabels=col, yticklabels=col, fmt='.5g')
    ax.set_ylabel("From")
    ax.set_xlabel("To")
    plt.title('Normal Control Mean Transition Probability')
    plt.savefig('./nvi_stats/NC_tpm.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(facecolor='white')
    ax = sns.heatmap(is_data.mean(axis=0), annot=True, cmap="Blues", xticklabels=col, yticklabels=col, fmt='.5g')
    ax.set_ylabel("From")
    ax.set_xlabel("To")
    plt.title('Infantile Epileptic Spasms Mean Transition Probability')
    plt.savefig('./nvi_stats/IS_tpm.png', dpi=300, bbox_inches='tight')
    plt.close()

    ms_num = nor_data.shape[-1]
    tp_hm = np.zeros((ms_num, ms_num))
    tp_hm_f = np.zeros((ms_num, ms_num))
    for i in range(ms_num):
        for j in range(ms_num):
            if i == j:
                continue
            else:
                nor_temp = nor_data[:, i, j]
                is_temp = is_data[:, i, j]
                f, p = scipy.stats.ttest_ind(nor_temp, is_temp)
                tp_hm[i, j] = p
                tp_hm_f[i, j] = f
    tp_hm = scipy.stats.false_discovery_control(tp_hm.reshape(-1)).reshape(ms_num, ms_num)
    c_list = []
    fv_list = []
    pv_list = []
    ast_list = []
    for i, r in enumerate(col):
        for j, c in enumerate(col):
            if r == c:
                continue
            else:
                fv_list.append(tp_hm_f[i, j])
                pv_list.append(tp_hm[i, j])
                c_list.append('{}_to_{}'.format(r, c))
    for pv in pv_list:
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

    plt.figure(facecolor='white')
    ax = sns.heatmap(tp_hm, annot=True, cmap="Blues", xticklabels=col, yticklabels=col, fmt='.5g')
    ax.set_ylabel("From")
    ax.set_xlabel("To")
    plt.title('Individual t-test result between NC and IS (p-value)')
    plt.savefig('./nvi_stats/tpm.png', dpi=300, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame()
    df['feature'] = c_list
    df['t-stat'] = fv_list
    df['p-value'] = pv_list
    df['asterisk'] = ast_list
    df.to_csv('./nvi_stats/tpm_stats.csv', encoding='utf-8')


def nvi_stats(data, label):
    data['Group'] = label
    normal = data.loc[data['Group'] == 'Normal'].drop('Group', axis=1)
    IS = data.loc[data['Group'] == 'ES'].drop('Group', axis=1)
    pv_list = []
    fv_list = []
    c_list = []
    for c in normal.columns:
        f, p = scipy.stats.ttest_ind(normal[c].values, IS[c].values)
        pv_list.append(p)
        fv_list.append(f)
        c_list.append(c)
    pv_list = scipy.stats.false_discovery_control(pv_list)
    df = pd.DataFrame()
    df['feature'] = c_list
    df['t-stat'] = fv_list
    df['p-value'] = pv_list
    ast_list = []
    for pv in pv_list:
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
    df['asterisk'] = ast_list
    return df


def flatten(data, label):
    f_data = []
    ms = []
    labels = []
    for i in range(data.shape[0]):
        temp = data.iloc[i]
        f_data.extend(temp.values)
        ms.extend([j.split('_')[0] for j in temp.index])
        labels.extend([label.iloc[i] for j in range(temp.shape[0])])
    return pd.DataFrame({'data': np.array(f_data), 'ms': np.array(ms), 'Group': labels})

if __name__ == '__main__':
    with open('./feat_dict.pkl', 'rb') as f:
        data = pickle.load(f)
    tp_data, tp_label, tp_col = nor_vs_is_tp(data)
    tpm_stats(tp_data, tp_label, tp_col)
    nvi = nor_vs_is(data)
    mean_corr = [i for i in nvi.columns if 'mean_corr' in i]
    gev = [i for i in nvi.columns if 'gev' in i]
    occur = [i for i in nvi.columns if 'occurrences' in i]
    timecov = [i for i in nvi.columns if 'timecov' in i]
    meandurs = [i for i in nvi.columns if 'meandurs' in i]
    nvi['Group'].loc[nvi['Group'] == 'IS'] = 'IESS'
    nvi['Group'].loc[nvi['Group'] == 'Normal'] = 'HC'
    mean_corr_df = flatten(nvi[mean_corr], nvi['Group'])
    gev_df = flatten(nvi[gev], nvi['Group'])
    occur_df = flatten(nvi[occur], nvi['Group'])
    timecov_df = flatten(nvi[timecov], nvi['Group'])
    meandurs_df = flatten(nvi[meandurs], nvi['Group'])

    pairs = [(('A', 'Normal'), ('A', 'ES')),
             (('B', 'Normal'), ('B', 'ES')),
             (('C', 'Normal'), ('C', 'ES')),
             (('D', 'Normal'), ('D', 'ES')),
             (('E', 'Normal'), ('E', 'ES')),
             ]
    fontdict = {'fontweight': 'bold', 'fontsize': 12}

    plt.figure(facecolor='white')
    colors = ('gainsboro', 'dimgrey')
    hue_order = ['HC', 'IESS']
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(data=gev_df, x='ms', y='data', hue='Group', hue_order=hue_order, fliersize=2.0)
    plt.xlabel('Microstates', fontdict=fontdict)
    plt.ylabel('GEV (%)', fontdict=fontdict)
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.savefig('./nvi_stats/gev.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(facecolor='white')
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(data=occur_df, x='ms', y='data', hue='Group', hue_order=hue_order, fliersize=2.0)
    plt.xlabel('Microstates', fontdict=fontdict)
    plt.ylabel('Occurrence (Hz)', fontdict=fontdict)
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.savefig('./nvi_stats/ouccur.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(facecolor='white')
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(data=timecov_df, x='ms', y='data', hue='Group', hue_order=hue_order, fliersize=2.0)
    plt.xlabel('Microstates', fontdict=fontdict)
    plt.ylabel('Time coverage (%)', fontdict=fontdict)
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.savefig('./nvi_stats/cov.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(facecolor='white')
    sns.set_palette(sns.color_palette(colors))
    ax = sns.boxplot(data=meandurs_df, x='ms', y='data', hue='Group', hue_order=hue_order, fliersize=2.0)
    plt.xlabel('Microstates', fontdict=fontdict)
    plt.ylabel('Mean durations (sec)', fontdict=fontdict)
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.savefig('./nvi_stats/mean_durs.png', dpi=300, bbox_inches='tight')
    plt.close()

    mean_corr_stats = nvi_stats(nvi[mean_corr], nvi['Group'])
    gev_stats = nvi_stats(nvi[gev], nvi['Group'])
    occur_stats = nvi_stats(nvi[occur], nvi['Group'])
    timecov_stats = nvi_stats(nvi[timecov], nvi['Group'])
    meandurs_stats = nvi_stats(nvi[meandurs], nvi['Group'])
    nvi_stats_df = pd.concat([mean_corr_stats, gev_stats, occur_stats,
                              timecov_stats, meandurs_stats], axis=0).reset_index(drop=True)
    nvi_stats_df.to_csv('./nvi_stats/nvi_stats.csv', encoding='utf-8')