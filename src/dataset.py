import mat73
import mne
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.signal import welch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from minirocket import fit, transform


# autoreject


def onehotEncoding(all_labels: list, ham1: str, ham2: str):
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
        print(all_labels)
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


def plot_psd(data, sfreq):
    # Compute PSD using Welch's method
    freqs, psds = welch(data, sfreq, nperseg=1920 / 8, noverlap=1024 / 8, axis=1)
    print(freqs.shape)
    print(psds.shape)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psds.T)
    # Plotting on a logarithmic scale for the y-axis
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.title("PSD using Welch Method")
    plt.legend(["ch" + str(i) for i in range(1, psds.shape[0] + 1)])
    plt.grid()
    plt.tight_layout()
    plt.show()


def compute_psd(data):
    info = mne.create_info(
        ch_names=["ch" + str(i) for i in range(1, 15)], sfreq=2400, ch_types="eeg"
    )
    raw = mne.io.RawArray(data, info)
    raw.filter(0.5, 50)
    psds, freqs = mne.time_frequency.psd_array_multitaper(
        data, sfreq=raw.info["sfreq"], fmin=0.5, fmax=50
    )
    avg_psds = np.mean(psds, axis=1)
    return avg_psds


def plot_single_scenario2(eeg_data):
    plt.figure(figsize=(15, 7))  # Adjust the figure size for better visualization

    for i in range(eeg_data.shape[0]):
        plt.plot(eeg_data[i, :], label=f"Channel {i+1}")

    plt.title("EEG Channels")
    plt.legend()  # To display the labels of each EEG channel
    plt.tight_layout()
    plt.show()


def plot_single_scenario(eeg_data):
    plt.figure(figsize=(15, 15))
    for i in range(14):
        plt.subplot(14, 1, i + 1)
        plt.plot(eeg_data[i, :])
        plt.title(f"Channel {i+1}")
        plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    EEG_PATH = "../datasets/Dasps.mat/"
    LABEL_PATH = "../datasets/participant_rating_public.xlsx"
    dataset = DASPSDataset(
        EEG_PATH, LABEL_PATH, range(1, 24), flag_fft=False, flag_min_rocket=True
    )
    print(len(dataset))  # 2 subjects * 6 situations = 12
    eeg, label = dataset[0]  # 0 - 5
    print(eeg.shape)
    print(label.shape)
    print(label)
    exit(1)
