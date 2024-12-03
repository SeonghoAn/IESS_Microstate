import numpy as np
from pycrostates.cluster import ModKMeans
import pickle
from tqdm import tqdm
from pycrostates.preprocessing import extract_gfp_peaks, resample
from pycrostates.io import ChData

col_names = ['A_mean_corr', 'A_gev', 'A_occurrences', 'A_timecov', 'A_meandurs', 'B_mean_corr', 'B_gev',
             'B_occurrences', 'B_timecov', 'B_meandurs', 'C_mean_corr', 'C_gev', 'C_occurrences', 'C_timecov',
             'C_meandurs', 'D_mean_corr', 'D_gev', 'D_occurrences', 'D_timecov', 'D_meandurs', 'F_mean_corr', 'F_gev',
             'F_occurrences', 'F_timecov', 'F_meandurs', 'unlabeled']

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
    nl_list = ['aaa'] # data is not available
    is_list = ['bbb'] # data is not available
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
    ModK.plot()
    return ModK


def ms_feature_per_subj(group, cluster):
    ms_data = {}
    for subj, data in tqdm(group.items()):
        data.load_data().drop_bad(verbose=False)
        segmentation = cluster.predict(data, factor=10, half_window_size=8)
        d = segmentation.compute_parameters()
        d['transition_matrix'] = {'tm': segmentation.compute_transition_matrix(),
                                  'cluster_name': segmentation.cluster_names}
        ms_data[subj] = d
    return ms_data


if __name__ == '__main__':
    with open("./epochs.pkl", 'rb') as f:
        data = pickle.load(f)
    data = rename_dict(data)
    data = remove_patients(data)
    normal = group_cluster(data['Normal'], 5) # num of clusters is found by cluster_sel.py.
    IS = group_cluster(data['IS'], 5)
    normal.invert_polarity([True, True, False, False, False])
    normal.reorder_clusters(order=[4, 0, 2, 1, 3])
    normal.rename_clusters(new_names=["A", "B", "C", "D", 'E'])
    IS.invert_polarity([True, False, False, False, True])
    IS.reorder_clusters(order=[4, 0, 1, 3, 2])
    IS.rename_clusters(new_names=["A", "B", "C", "D", 'E'])
    normal_feat = ms_feature_per_subj(data['Normal'], normal)
    is_feat = ms_feature_per_subj(data['IS'], normal)
    feat_dict = {'Normal': normal_feat, 'IS': is_feat}
    with open('./feat_dict.pkl', 'wb') as f:
        pickle.dump(feat_dict, f)