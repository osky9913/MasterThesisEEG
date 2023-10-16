import mne,pywt
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import autoreject
import tensorflow as tf
from sklearn.cross_decomposition import CCA



def autoReject(raw,duration=15):
    epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True)
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                               n_jobs=1, verbose=True)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    print(epochs_ar)


def wavelet_denoising(data, wavelet='db4', level=1):
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1 / 0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


def plot_cleaning(raw, raw_corrected, name,situation):
    output_dir = os.path.join('data', name)
    os.makedirs(output_dir, exist_ok=True)

    output_dir2 = os.path.join(output_dir, situation)
    os.makedirs(output_dir2, exist_ok=True)

    print(output_dir2)

    with mne.viz.use_browser_backend('matplotlib'):
        # plt.figure(figsize=(12, 8))
        raw.plot(n_channels=14, duration=15, title="Before", show=False)
        plt.savefig((os.path.join(output_dir2, 'before.png')))
        plt.close()

        raw_corrected.plot(n_channels=14, duration=15, title="After", show=False)
        plt.savefig((os.path.join(output_dir2, 'after.png')))
        plt.close()

    plt.figure(figsize=(12, 8))

    offsets = np.arange(raw._data.shape[0]) * 0.00009
    plt.figure(figsize=(12, 8))

    # Plot original data in red and cleaned data in blue
    for i, (ch_data, ch_data_cleaned) in enumerate(zip(raw._data, raw_corrected._data)):
        plt.plot(raw.times, ch_data + offsets[i], color='red', lw=0.5)
        plt.plot(raw.times, ch_data_cleaned + offsets[i], color='blue', lw=0.5)

    plt.yticks(offsets, raw.ch_names)
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')
    plt.title('EEG Data: Original (Red) vs Cleaned (Blue)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir2, 'combined_plot.png'))


def extract_robust_features(eeg_data, fs=128):
    """
    Extract robust features from multi-channel EEG signals for deep learning.

    Parameters:
    - eeg_data: A 2D numpy array representing the EEG signals with shape (channels, time_points)
    - fs: Sampling rate

    Returns:
    - features_array: A 1D numpy array containing the extracted features
    """

    num_channels, _ = eeg_data.shape
    all_features = []

    eeg_bands = {'Delta': (0.5, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 13),
                 'Beta': (13, 30),
                 'Gamma': (30, 40)}

    for i in range(num_channels):
        channel_data = eeg_data[i]

        # Time-domain features
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        skw = skew(channel_data)
        kurt = kurtosis(channel_data)
        all_features.extend([mean, variance, skw, kurt])

        # Frequency-domain features using Welch method
        freqs, psd = welch(channel_data, fs=fs)
        for band, freq_range in eeg_bands.items():
            idx = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
            band_power = np.sum(psd[idx])
            all_features.append(band_power)

    """    
    # Inter-channel coherence (taking Alpha band as an example)
    for i in range(num_channels):
        for j in range(i+1, num_channels):
            f, Cxy = coherence(eeg_data[i], eeg_data[j], fs=fs)
            idx = np.where((f >= eeg_bands['Alpha'][0]) & (f <= eeg_bands['Alpha'][1]))[0]
            mean_coherence = np.mean(Cxy[idx])
            all_features.append(mean_coherence)
    """
    features_array = np.array(all_features)
    return features_array


def clean_eeg(eeg_signal,
              filter_flag,
              wavelet_flag,
              ica_flag,
              ica_method,
              l_freq, h_freq):
    """
    :param eeg_signal: (14,1920 ) -> (Channels,timeBasedValues)
    :param filter_flag:
    :param wavelet_flag:
    :param ica_flag:
    :param ica_method: 'fastica’ | ‘infomax’ | ‘picard’
    :param l_freq: 1
    :param h_freq: None
    :return:
    """
    # print("subject: ",subject)
    # print("subject.shape: ",subject.shape)
    info = mne.create_info(
        ch_names=['AF3', 'AF4', 'F3', 'F4', 'FC5', 'FC6', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'O1', 'O2'],
        sfreq=128,
        ch_types='eeg',
    )
    montage = mne.channels.make_standard_montage('standard_1020')
    raw = mne.io.RawArray(eeg_signal, info)
    mne.set_log_level(verbose="WARNING")
    raw.set_montage(montage)





    # Apply wavelet denoising
    # raw._data = np.array([wavelet_denoising(chan) for chan in raw._data])
    raw_corrected = raw.copy()
    if filter_flag:
        raw_corrected.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', filter_length='auto')

    if wavelet_flag:
        raw_corrected._data = np.array([wavelet_denoising(chan) for chan in raw_corrected._data])

    if ica_flag:
        # ICA for artifact removal
        ica = mne.preprocessing.ICA(method=ica_method, fit_params={"extended": True}, random_state=1)
        ica.fit(raw_corrected)
        ica.apply(raw_corrected)

    #autoReject(raw_corrected)
    return raw._data, raw_corrected._data

def convert_pytorch_dataset_to_tf_dataset(pytorch_dataset):
    # Initialize lists to store converted data
    tf_features = []
    tf_labels = []

    # Iterate through the PyTorch dataset and convert samples
    for pytorch_sample in pytorch_dataset:
        # Extract features and labels from PyTorch sample
        features, labels = pytorch_sample

        # Convert PyTorch tensors to TensorFlow tensors
        tf_features.append(tf.convert_to_tensor(features.numpy()))
        tf_labels.append(tf.convert_to_tensor(labels.numpy()))

    #tf_dataset = tf.data.Dataset.from_tensor_slices((tf_features, tf_labels))
    tf_features = tf.convert_to_tensor(tf_features)
    tf_labels = tf.convert_to_tensor(tf_labels)
    return tf_features, tf_labels
    """
    # Identify artifact components based on correlation with frontal EEG channels

    eye_chans = ['AF3', 'AF4', 'F7', 'F8']
    eog_inds = []
    for ch in eye_chans:
        eog_inds_temp, scores_temp = ica.find_bads_eog(raw_corrected, ch_name=ch, l_freq=1, h_freq=15)
        eog_inds.extend(eog_inds_temp)

    # Ensure unique indices
    eog_inds = list(set(eog_inds))

    ica.exclude = eog_inds
    """
    # Apply the ICA to remove the artifacts

    # Visualization of the raw and corrected data





