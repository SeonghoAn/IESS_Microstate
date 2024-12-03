import mne
import glob
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser(description='WS_data_generation')
parser.add_argument('--file_name', type=str, default='epochs')
args = vars(parser.parse_args())

mapping = {'EEG Fp1-REF':'Fp1', 'EEG F7-REF':'F7', 'EEG T3-REF':'T3',
           'EEG T5-REF':'T5', 'EEG Fp2-REF':'Fp2', 'EEG F8-REF':'F8',
           'EEG T4-REF':'T4', 'EEG T6-REF':'T6', 'EEG F3-REF':'F3',
           'EEG C3-REF':'C3', 'EEG P3-REF':'P3', 'EEG O1-REF':'O1',
           'EEG F4-REF':'F4', 'EEG C4-REF':'C4', 'EEG P4-REF':'P4',
           'EEG O2-REF':'O2', 'EEG Fz-REF':'Fz', 'EEG Cz-REF':'Cz',
           'EEG Pz-REF':'Pz'}

mapping_modif = {'EEG Fp1-REF':'Fp1', 'EEG F7-REF':'F7', 'EEG T7-REF':'T3',
           'EEG P7-REF':'T5', 'EEG Fp2-REF':'Fp2', 'EEG F8-REF':'F8',
           'EEG T8-REF':'T4', 'EEG P8-REF':'T6', 'EEG F3-REF':'F3',
           'EEG C3-REF':'C3', 'EEG P3-REF':'P3', 'EEG O1-REF':'O1',
           'EEG F4-REF':'F4', 'EEG C4-REF':'C4', 'EEG P4-REF':'P4',
           'EEG O2-REF':'O2', 'EEG Fz-REF':'Fz', 'EEG Cz-REF':'Cz',
           'EEG Pz-REF':'Pz'}

def NL_data_loader(subject_id):
    data_path = './data/NL/HarmEEG/'
    path_dict = {}
    for i in subject_id:
        path_1 = glob.glob(data_path+"*"+str(i)+'*')
        path_2 = glob.glob(path_1[0]+"/*.edf")
        path_dict[i] = path_2[0]
    return path_dict

def IS_data_loader(subject_id):
    data_path = './data/IS/'
    path_dict = {}
    for i in subject_id:
        path_1 = glob.glob(data_path+"*"+str(i)+'*')
        if i in ['aaa', 'bbb', 'ccc ']: # Data is not available.
            path_dict[i] = path_1[0]
        else:
            path_2 = glob.glob(path_1[0]+"/*.edf")
            path_dict[i] = path_2[0]
    return path_dict


def IS_preprocesser(path_dict, subject_id, label):
    class_0_dict = dict()
    class_1_dict = dict()
    for idx in tqdm(subject_id):
        path = path_dict[idx]
        raw = mne.io.read_raw_edf(path, verbose=False)
        drop = []
        for i in raw.ch_names:
            if ('EEG' not in i) or ('T1' in i) or ('T2' in i) or ('A1' in i) or ('A2' in i):
                drop.append(i)
        raw.drop_channels(drop)
        if 'EEG T3-REF' in raw.ch_names:
            mne.rename_channels(raw.info, mapping)
        else:
            mne.rename_channels(raw.info, mapping_modif)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        raw.load_data(verbose=False)
        raw.set_eeg_reference('average', verbose=False)
        raw.notch_filter(60.0, verbose=False)
        raw.filter(l_freq=2.0, h_freq=20.0, verbose=False)
        ica = mne.preprocessing.ICA(
            n_components=0.95, method="picard", max_iter="auto", random_state=42
        )
        ica.fit(raw, verbose=False)
        muscle_idx_auto, scores = ica.find_bads_muscle(raw, verbose=False)
        ica.apply(raw, exclude=muscle_idx_auto, verbose=False)
        epoch = mne.make_fixed_length_epochs(raw, duration=30.0, verbose=False)
        y = label[idx]
        if y == 0:
            class_0_dict[idx] = epoch
        elif y == 1:
            class_1_dict[idx] = epoch

    return class_0_dict, class_1_dict


def NL_preprocesser(path_dict, subject_id):
    class_2_dict = dict()
    for idx in tqdm(subject_id):
        path = path_dict[idx]
        raw = mne.io.read_raw_edf(path, verbose=False)
        drop = []
        for i in raw.ch_names:
            if ('EEG' not in i) or ('T1' in i) or ('T2' in i) or ('A1' in i) or ('A2' in i):
                drop.append(i)
        raw.drop_channels(drop)
        if 'EEG T3-REF' in raw.ch_names:
            mne.rename_channels(raw.info, mapping)
        else:
            mne.rename_channels(raw.info, mapping_modif)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        raw.load_data(verbose=False)
        raw.set_eeg_reference('average', verbose=False)
        raw.notch_filter(60.0, verbose=False)
        raw.filter(l_freq=2.0, h_freq=20.0, verbose=False)
        ica = mne.preprocessing.ICA(
            n_components=0.95, method="picard", max_iter="auto", random_state=42
        )
        ica.fit(raw, verbose=False)
        muscle_idx_auto, scores = ica.find_bads_muscle(raw, verbose=False)
        ica.apply(raw, exclude=muscle_idx_auto, verbose=False)
        epoch = mne.make_fixed_length_epochs(raw, duration=30.0, verbose=False)
        class_2_dict[idx] = epoch

    return class_2_dict

def label_loader(label_path):
    # label_path = "./Subject_info.csv"
    label_df = pd.read_csv(label_path)
    subject_id = label_df['Name']
    label = dict()
    # label = label_df['Intractable epilepsy = 1']
    for idx in subject_id:
        label[idx] = label_df.loc[label_df['Name']==idx]['Intractable epilepsy = 1'].values[0]
    return subject_id, label

def mne_loader():
    label_path = "Subject_info.csv"
    subject_id, label = label_loader(label_path)
    IS_path_dict = IS_data_loader(subject_id)
    subject_id_nl = ['aaa'] # Data is not available.
    NL_path_dict = NL_data_loader(subject_id_nl)
    class_0_dict, class_1_dict = IS_preprocesser(IS_path_dict, subject_id, label)
    class_2_dict = NL_preprocesser(NL_path_dict, subject_id_nl)
    output = {'Responder': class_0_dict, 'Non-responder': class_1_dict, 'Normal': class_2_dict}
    return output

def main(file_name):
    data = mne_loader()
    with open('./{}.pkl'.format(file_name), 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main(args['file_name'])