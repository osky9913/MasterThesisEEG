import torch
import mat73
import mne
import numpy as np
import torch.nn as nn
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from minirocket import fit, transform
from mne.preprocessing import ICA
import seaborn as sns


def apply_filter_ica(epochs_eeg):
    epochs_eeg.filter(l_freq=1.0, h_freq=None)

    # Fit ICA
    ica = ICA(n_components=14, random_state=97)
    ica.fit(epochs_eeg)

    # Make a copy of the original epochs for cleaning
    epochs_clean = epochs_eeg.copy()

    # Apply ICA to the copied epochs
    ica.apply(epochs_clean)
    return epochs_clean

def onehotEncoding(all_labels: list, ham1: str, ham2: str):
    encoding = [0] * len(all_labels)
    if ham1 in all_labels:
        index = all_labels.index(ham1)
        encoding[index] = 1
    if ham2 in all_labels:
        index = all_labels.index(ham2)
        encoding[index] = 1
    return encoding

def onehotEncoding2(all_labels: list, ham1: str, ham2: str):
    encoding = [0] * len(all_labels)
    if ham1 in all_labels:
        index = all_labels.index(ham1)
        encoding[index] = 1
    if ham2 in all_labels:
        index = all_labels.index(ham2)
        encoding[index] = 1
    return encoding


class DASPSDataset(Dataset):
    def __init__(
        self,
        eeg_path,
        labels_path,
        subject_range,
        flag_psd=False,
        flag_fft=False,
        flag_min_rocket=False,
        device="cpu",
    ):
        self.eeg_path = eeg_path
        self.labels_path = labels_path
        self.subject_range = subject_range
        self.flag_psd = flag_psd
        self.flat_fft = flag_fft
        self.flag_min_rocket = flag_min_rocket
        self.device = device

        self.data = {}
        self.data_transformed = []
        self.encoder = LabelEncoder()
        self.dataset = []

        # Load EEG data
        for i in self.subject_range:
            id_subject = f"S{i:02d}"
            mat_file = self.eeg_path + id_subject + "preprocessed.mat"
            data = mat73.loadmat(mat_file)["data"]
            self.data[f"{id_subject}"] = data.transpose(2, 0, 1)

        # Load and encode labels
        df = pd.read_excel(self.labels_path)
        all_labels = (
            df["Hamilton1"].astype(str).tolist() + df["Hamilton2"].astype(str).tolist()
        )
        all_labels = [x.split(":")[1] for x in all_labels if x != "nan"]
        all_labels = list(set(all_labels))
        self.encoder.fit(all_labels)
        self.all_labels = all_labels
        # self.binary_labels = convert_to_binary(self.encoder, all_labels)

        for i in self.subject_range:
            id_subject = f"S{i:02d}"
            subject_rows = df[df["Id Participant"] == id_subject]
            for index, row in subject_rows.iterrows():
                hamilton1 = row["Hamilton1"].split(":")[1]
                hamilton2 = row["Hamilton2"].split(":")[1]
                labels = onehotEncoding(all_labels, hamilton1, hamilton2)
                # labels = self.encoder.transform([hamilton1, hamilton2])
                # Use the pre-converted binary labels

                self.data[f"{id_subject}"] = (self.data[f"{id_subject}"], labels)

        for key in self.data.keys():
            situations, labels_values = self.data[key]
            for situation in range(situations.shape[0]):
                x = situations[situation]
                y = labels_values
                if flag_fft:
                    x = np.fft.fft(x, axis=1)
                if flag_min_rocket:
                    feat = fit(x, 400)
                    x = transform(x, feat)
                self.dataset.append((x, y))

    def get_labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        eeg_data = torch.tensor(
            self.dataset[idx][0], dtype=torch.float32, device=self.device
        )
        labels = torch.tensor(
            self.dataset[idx][1], dtype=torch.float32, device=self.device
        )
        return eeg_data, labels







def clean_eeg(subject):
    print(subject)
    data_reshaped = subject.reshape(14, -1)
    print(data_reshaped)
    return
    # Create an MNE Info object
    info = mne.create_info(
        ch_names=['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2'],
        sfreq=128,
        ch_types='eeg',
    )

    # Convert the data to a Raw object
    raw = mne.io.RawArray(data_reshaped, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Now you can filter, compute and apply SSP, or perform ICA just like with Epochs

    # For example, filtering
    raw.filter(l_freq=1.0, h_freq=None)

    # Compute and apply ICA
    ica = ICA(n_components=14, random_state=97)
    ica.fit(raw)

    # Create a copy to apply ICA
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # Plot
    raw.plot(start=0, duration=15, title="Original Data", n_channels=14)
    raw_clean.plot(start=0, duration=15, title="Cleaned Data", n_channels=14)
    plt.show()
    """
    eeg_data = np.transpose(subject, (2, 0, 1))

    # Create an MNE Info object
    info = mne.create_info(
        ch_names=['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2'],
        sfreq=128,
        ch_types='eeg',
    )
    montage = mne.channels.make_standard_montage('standard_1020')


    # Create an events array
    """
    """
    n_situations = eeg_data.shape[0]
    events = np.column_stack([
        np.arange(0, n_situations * 1920, 1920),
        np.zeros(n_situations, dtype=int),
        np.arange(1, n_situations + 1)
    ])
    """
    """
    epochs_a = mne.EpochsArray(eeg_data, info, tmin=0)
    epochs_b = mne.EpochsArray(eeg_data, info, tmin=0)

    # Set the montage
    epochs_a.set_montage(montage)
    epochs_b.set_montage(montage)
    epochs_a.compute_psd(fmax=50).plot(picks="data", exclude="bads")
    epochs_a.plot(duration=5, n_channels=14)



    epochs_b = apply_filter_ica(epochs_b)
    blink_projs, _ = mne.preprocessing.compute_proj_eog(epochs_b, n_eeg=1, ch_name='AF4')

    # Add the SSP projectors to the epochs
    epochs_b.add_proj(blink_projs)

    # Now when you plot the data, you can specify to apply the SSP projectors
    epochs_b.average().plot(proj=True)

    """
    return
    # Filter the data

    print("printing data from epochs a ")
    print(epochs_a.get_data())

    # Select an epoch index for plotting (e.g., the first epoch)
    epoch_idx = 0

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Plot sensor data before cleaning
    plt.subplot(2, 2, 1)
    epochs_a[epoch_idx].plot_image(picks='eeg', show=False)
    plt.title('Original EEG Data')

    # Plot sensor data after cleaning
    plt.subplot(2, 2, 2)
    epochs_b[epoch_idx].plot_image(picks='eeg', show=False)
    plt.title('Cleaned EEG Data')

    # Plot power spectral density before cleaning
    plt.subplot(2, 2, 3)
    epochs_a[epoch_idx].plot_psd(picks='eeg', show=False)
    plt.title('Original PSD')

    # Plot power spectral density after cleaning
    plt.subplot(2, 2, 4)
    epochs_b[epoch_idx].plot_psd(picks='eeg', show=False)
    plt.title('Cleaned PSD')

    # Show the plot
    plt.tight_layout()
    plt.show()


class DASPSDataset2(Dataset):
    def __init__(
            self,
            eeg_path,
            labels_path,
            subject_range,
            flag_psd=False,
            flag_fft=False,
            flag_min_rocket=False,
            device="cpu",
    ):
        self.eeg_path = eeg_path
        self.labels_path = labels_path
        self.subject_range = subject_range
        self.flag_psd = flag_psd
        self.flat_fft = flag_fft
        self.flag_min_rocket = flag_min_rocket
        self.device = device

        self.data = {}
        self.data_transformed = []
        self.encoder = LabelEncoder()
        self.dataset = []

        # Load EEG data
        for i in self.subject_range:
            id_subject = f"S{i:02d}"
            mat_file = self.eeg_path + id_subject + "preprocessed.mat"
            loaded_mat_file = mat73.loadmat(mat_file)
            data = loaded_mat_file["data"]
            eeg_data = data
            clean_eeg(eeg_data)

            # experiments

            # epochsA.plot_projs_topomap(vlim="joint")
            """
            # Select one channel for plotting (e.g., the first channel)
            channel_idx = 0

            # Create a time vector for plotting
            times = np.arange(data_original.shape[1]) / info['sfreq']

            # Plot original and cleaned data on the same plot
            plt.figure(figsize=(10, 6))
            plt.plot(times, data_clean[channel_idx, :], label='Cleaned')

            plt.plot(times, data_original[channel_idx, :], label='Original')
            plt.legend()
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (uV)')
            plt.title(f'EEG Channel {info["ch_names"][channel_idx]} - Epoch {epoch_idx + 1}')
            plt.show()
            """
            return
        # Load and encode labels
        df = pd.read_excel(self.labels_path)
        all_labels = (
                df["Hamilton1"].astype(str).tolist() + df["Hamilton2"].astype(str).tolist()
        )
        all_labels = [x.split(":")[1] for x in all_labels if x != "nan"]
        all_labels = list(set(all_labels))
        self.encoder.fit(all_labels)
        self.all_labels = all_labels
        # self.binary_labels = convert_to_binary(self.encoder, all_labels)

        for i in self.subject_range:
            id_subject = f"S{i:02d}"
            subject_rows = df[df["Id Participant"] == id_subject]
            for index, row in subject_rows.iterrows():
                hamilton1 = row["Hamilton1"].split(":")[1]
                hamilton2 = row["Hamilton2"].split(":")[1]
                labels = onehotEncoding(all_labels, hamilton1, hamilton2)
                # labels = self.encoder.transform([hamilton1, hamilton2])
                # Use the pre-converted binary labels

                self.data[f"{id_subject}"] = (self.data[f"{id_subject}"], labels)

        for key in self.data.keys():
            situations, labels_values = self.data[key]
            for situation in range(situations.shape[0]):
                x = situations[situation]
                y = labels_values
                if flag_fft:
                    x = np.fft.fft(x, axis=1)
                if flag_min_rocket:
                    feat = fit(x, 400)
                    x = transform(x, feat)
                self.dataset.append((x, y))

    def get_labels(self):
        return self.all_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        eeg_data = torch.tensor(
            self.dataset[idx][0], dtype=torch.float32, device=self.device
        )
        labels = torch.tensor(
            self.dataset[idx][1], dtype=torch.float32, device=self.device
        )
        return eeg_data, labels



"""
print("23 subjects * 6 situations * 2 = 276 ==", len(dataset))  # 23 subjects * 6 situations * 2 = 276
eeg_feature, label = dataset[0]  # 
print("eeg_feature.shape:", eeg_feature.shape)
print("eeg_feature: ",eeg_feature)
print("label.shape:",label.shape)
print("label:", label)
print("all labels:",dataset.get_labels())
"""

if __name__ == "__main__":
    EEG_PATH = "../datasets/Dasps.mat/"
    LABEL_PATH = "../datasets/participant_rating_public.xlsx"
    dataset = DASPSDataset2(EEG_PATH, LABEL_PATH, range(1, 24), flag_fft=False, flag_min_rocket=True)
