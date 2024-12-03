from pycrostates.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    dunn_score,
    davies_bouldin_score,
)
import numpy as np
from pycrostates.cluster import ModKMeans
import pickle
from tqdm import tqdm
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.io import ChData
import matplotlib.pyplot as plt
import copy

plt.rcParams['font.family'] = 'Times New Roman'


def rename_dict(data):
    rename_dict = {}
    temp = data['Responder'] | data['Non-responder']
    rename_dict['IS'] = temp
    rename_dict['Normal'] = data['Normal']
    return rename_dict


def remove_patients(data):
    rm_dict = {}
    nl_dict = {}
    is_dict = {}
    nl_list = ['aaa']  # data is not available
    is_list = ['bbb']  # data is not available
    for k in data['Normal'].keys():
        if k in nl_list:
            print('Remove', k)
            continue
        else:
            nl_dict[k] = data['Normal'][k]

    for k in data['IS'].keys():
        if k in is_list:
            print('Remove', k)
            continue
        else:
            is_dict[k] = data['IS'][k]
    rm_dict['Normal'] = nl_dict
    rm_dict['IS'] = is_dict
    return rm_dict


def group_cluster(group, n_clusters):
    individual_cluster_centers = list()
    for subj in tqdm(group.values()):
        subj.drop_bad(verbose=False)
        gfp_peaks = extract_gfp_peaks(subj)
        ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
        ModK.fit(gfp_peaks, n_jobs=-1)
        individual_cluster_centers.append(ModK.cluster_centers_)
    group_cluster_centers = np.vstack(individual_cluster_centers).T
    group_cluster_centers = ChData(group_cluster_centers, ModK.info)
    ModK = ModKMeans(n_clusters=n_clusters, random_state=42)
    ModK.fit(group_cluster_centers, n_jobs=-1)
    return ModK


def evaluation(data):
    cluster_numbers = range(4, 11)
    scores = {
        "Silhouette": np.zeros(len(cluster_numbers)),
        "Calinski-Harabasaz": np.zeros(len(cluster_numbers)),
        "Dunn": np.zeros(len(cluster_numbers)),
        "Davies-Bouldin": np.zeros(len(cluster_numbers)),
    }
    for k, n_clusters in enumerate(cluster_numbers):
        print('=' * 100)
        print('Extract cluster: {}'.format(n_clusters))
        # fit K-means algorithm with a set number of cluster centers
        ModK = group_cluster(data, n_clusters)
        #         ModK = ModKMeans(n_clusters=4, random_state=42)
        #         ModK.fit(gfp_peaks, n_jobs=-1)
        # compute scores
        scores["Silhouette"][k] = silhouette_score(ModK)
        scores["Calinski-Harabasaz"][k] = calinski_harabasz_score(ModK)
        scores["Dunn"][k] = dunn_score(ModK)
        scores["Davies-Bouldin"][k] = davies_bouldin_score(ModK)
    return scores


def flatten(data):
    rename_dict = {}
    rename_dict['All'] = data['Normal']  # data['Responder'] | data['Non-responder'] |
    return rename_dict

def meta_criterion(rank):
    q75, q25 = np.percentile(rank, [75 ,25])
    iqr = q75 - q25
    iqr_idx = np.where((rank>=q25) & (rank<=q75))
    iqm = rank[iqr_idx].mean()
    return (iqm*iqm) / iqr

if __name__ == '__main__':
    with open("./epochs.pkl", 'rb') as f:
        data = pickle.load(f)

    data = rename_dict(data)
    data = remove_patients(data)
    scores = evaluation(data['Normal'])
    f, ax = plt.subplots(2, 2, sharex=True, facecolor='white')
    for k, (score, values) in enumerate(scores.items()):
        ax[k // 2, k % 2].bar(x=range(4, 11), height=values, color='grey')
        ax[k // 2, k % 2].set_title(score)
    plt.text(
        0.03, 0.5, "Score",
        horizontalalignment='center',
        verticalalignment='center',
        rotation=90,
        fontdict=dict(size=14),
        transform=f.transFigure,
    )
    plt.text(
        0.5, 0.03, "Number of clusters",
        horizontalalignment='center',
        verticalalignment='center',
        fontdict=dict(size=14),
        transform=f.transFigure,
    )
    plt.savefig('./results/cluster_eval.png', dpi=300, bbox_inches='tight')
    plt.close()

    r_scores = copy.deepcopy(scores)
    ranks = []
    for key, value in r_scores.items():
        if key != "Davies-Bouldin":
            ranks.append(list(sorted(value, reverse=False).index(ele) for ele in value))
        else:
            ranks.append(list(sorted(value, reverse=True).index(ele) for ele in value))
    ranks = np.array(ranks)
    avg_ranks = ranks.mean(axis=0)
    mc_ranks = [meta_criterion(ranks[:, i]) for i in range(ranks.shape[-1])]
    n_clusters = range(4, 11)
    plt.figure(facecolor='white')
    plt.bar(x=n_clusters, height=mc_ranks, color='grey')
    plt.xlabel('Number of clusters')
    plt.ylabel('Meta-criterion')
    plt.savefig('./results/meta_criterion.png', dpi=300, bbox_inches='tight')
    plt.close()
