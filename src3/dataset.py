import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from minirocket import fit, transform
import mat73
import pandas as pd
from utils import clean_eeg, extract_robust_features
import pprint


def one_hot_encoding_categorical(all_labels: list, ham1):
    """
    :param all_labels: list of values from excel
    :param ham1:
    :return:
    """
    encoding = [0] * len(all_labels)
    if ham1 in all_labels:
        index = all_labels.index(ham1)
        encoding[index] = 1

    return encoding


def one_hot_encoding_categorical_bce(all_labels: list, ham1, ham2):
    encoding1 = [0] * len(all_labels)
    encoding2 = [0] * len(all_labels)

    if ham1 in all_labels:
        index = all_labels.index(ham1)
        encoding1[index] = 1

    if ham2 in all_labels:
        index = all_labels.index(ham1)
        encoding2[index] = 1

    return encoding1  + encoding2


class DASPSDataset2(Dataset):
    def __init__(
            self,
            eeg_path,
            labels_path,
            subject_range,
            flag_categorical_enc,
            flag_categorical_bin,
            flag_categorical_bin_bce,
            flag_psd=False,
            flag_fft=False,
            flag_clean=False,
            flag_filter=True,
            flag_wavelet=False,
            flag_ica=True,
            ica_method='infomax',
            l_freq=1,
            h_freq=None,
            eeg_splitter_sec=1,
            flag_min_rocket=False,
            number_features_rocket=400,
            number_sps_robust=128,
            flag_new_feat=False,

            device="cpu",
    ):
        self.eeg_path = eeg_path
        self.labels_path = labels_path
        self.subject_range = subject_range
        self.flag_categorical_enc = flag_categorical_enc
        self.flag_categorical_bin = flag_categorical_bin
        self.flag_categorical_bin_bce = flag_categorical_bin_bce
        self.flag_psd = flag_psd
        self.flat_fft = flag_fft
        self.flag_clean = flag_clean
        self.flag_min_rocket = flag_min_rocket
        self.flag_new_feat = flag_new_feat
        self.device = device

        self.data = {}
        self.data_transformed = []
        self.encoder = LabelEncoder()
        self.dataset = []

        # Load EEG data
        for i in self.subject_range:
            id_subject = f"S{i:02d}"
            mat_file = self.eeg_path + id_subject + "preprocessed.mat"
            data = mat73.loadmat(mat_file)["data"] * 1e-6
            self.data[f"{id_subject}"] = dict()
            self.data[f"{id_subject}"]["raw"] = data.transpose(2, 0, 1)

        # Load and encode labels
        df = pd.read_excel(self.labels_path)
        all_labels = (
                df["Hamilton1"].astype(str).tolist() + df["Hamilton2"].astype(str).tolist()
        )
        all_labels = [x.split(":")[1] for x in all_labels if x != "nan"]
        all_labels = list(set(all_labels))
        self.all_labels = all_labels
        self.encoder.fit(all_labels)
        self.all_labels = all_labels
        # self.binary_labels = convert_to_binary(self.encoder, all_labels)

        for i in self.subject_range:
            id_subject = f"S{i:02d}"
            subject_rows = df[df["Id Participant"] == id_subject]
            for index, row in subject_rows.iterrows():
                hamilton1 = row["Hamilton1"].split(":")[1]
                hamilton2 = row["Hamilton2"].split(":")[1]
                label1 = one_hot_encoding_categorical(all_labels, hamilton1)
                label2 = one_hot_encoding_categorical(all_labels, hamilton2)
                categorical_bin = [label1, label2]
                categorical_enc = self.encoder.transform([hamilton1, hamilton2])
                categorical_bin_bce_loss = one_hot_encoding_categorical_bce(all_labels, hamilton1, hamilton2)
                # Use the pre-converted binary labels
                self.data[f"{id_subject}"]["categorical_bin"] = categorical_bin
                self.data[f"{id_subject}"]["categorical_enc"] = categorical_enc
                self.data[f"{id_subject}"]["categorical_bin_bce_loss"] = categorical_bin_bce_loss

        for key in self.data.keys():
            subject_values = self.data[key]
            situations = subject_values["raw"]
            labels_values = subject_values["categorical_bin"]
            encoder_values = subject_values["categorical_enc"]
            labels_bce_values = subject_values["categorical_bin_bce_loss"]
            for situation in range(situations.shape[0]):
                self.data[key]["situations"] = {}
                self.data[key]["situations"][str(situation)] = {}
                self.data[key]["situations"][str(situation)]["raw"] = subject_values["raw"][situation]
                if self.flag_clean:
                    raw, cleaned = clean_eeg(subject_values["raw"][situation],
                                             filter_flag=flag_filter,
                                             wavelet_flag=flag_wavelet,
                                             ica_flag=flag_ica,
                                             ica_method=ica_method,
                                             l_freq=l_freq,
                                             h_freq=h_freq)
                    self.data[key]["situations"][str(situation)]["clean"] = cleaned
                else:
                    self.data[key]["situations"][str(situation)]["clean"] = \
                    self.data[key]["situations"][str(situation)]["raw"]

                split_vectors_clean = np.split(self.data[key]["situations"][str(situation)]["clean"], eeg_splitter_sec,
                                         axis=1)

                split_vectors_raw = np.split(self.data[key]["situations"][str(situation)]["raw"], eeg_splitter_sec,
                                         axis=1)
                for vec_clean, vec_raw in zip(split_vectors_clean, split_vectors_raw):
                    row = {}
                    vec_clean = vec_clean.astype(np.float32)

                    feat = fit(vec_clean, number_features_rocket)
                    row['min_rocket'] = transform(vec_clean, feat).flatten()
                    row["robust"] = extract_robust_features(vec_clean, number_sps_robust)
                    row['categorical_bin'] = labels_values
                    row['categorical_enc'] = encoder_values
                    row["categorical_bin_bce_loss"] = labels_bce_values
                    row["clean"] = vec_clean
                    row["raw"] = vec_raw
                    self.dataset.append(row)

    def get_labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.dataset)

    def custom_get_item(self,idx):
        return self.dataset[idx]

    def __getitem__(self, idx):

        row = self.dataset[idx]
        x, y = None, None
        if self.flag_min_rocket:
            x = row["min_rocket"]

        if self.flag_new_feat:
            x = row["robust"]

        if self.flag_categorical_enc:
            y = row["categorical_enc"]

        if self.flag_categorical_bin:
            y = row["categorical_bin"]

        if self.flag_categorical_bin_bce:
            y = row["categorical_bin_bce_loss"]

        x = torch.tensor(
            x, dtype=torch.float32, device=self.device
        )

        y = torch.tensor(
            y, dtype=torch.float32, device=self.device
        )
        return x, y


if __name__ == "__main__":
    EEG_PATH = "../datasets/Dasps.mat/"
    LABEL_PATH = "../datasets/participant_rating_public.xlsx"
    dataset = DASPSDataset2(
        EEG_PATH, LABEL_PATH, range(1, 3),
        flag_categorical_bin=False,
        flag_categorical_enc=False,
        flag_categorical_bin_bce=True,
        flag_psd=False,
        flag_fft=False,
        flag_clean=True,
        flag_filter=True,
        flag_wavelet=False,
        flag_ica=True,
        ica_method='infomax',
        l_freq=1,
        h_freq=None,
        flag_min_rocket=False,
        number_features_rocket=100,
        number_sps_robust=128,
        flag_new_feat=True,
        device='cpu'
    )
    print(dataset.all_labels)
    print(len(dataset))  # 2 subjects * 6 situations = 12
    eeg, label = dataset[0]  # 0 - 5
    print(eeg.shape)
    print(label.shape)
    print(label)
    pprint.pprint(dataset.custom_get_item(0))

    exit(1)
