from scipy.io import loadmat
import mne


mat = loadmat("A01T.mat",simplify_cells=True)
eeg = mat["data"][3]["X"] * 1e-6 # convert to volts


ch_names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1",
"Cz","C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2",
"POz", "EOG1", "EOG2", "EOG3"]

info = mne.create_info(ch_names, 250, ch_types=["eeg"] * 22 +
["eog"] * 3)

raw = mne.io.RawArray(eeg.T, info)

raw.set_montage("standard_1020")

# Settuping ICA


raw_tmp = raw.copy()
raw_tmp.filter(l_freq=1, h_freq=None)
ica = mne.preprocessing.ICA(method="infomax", fit_params={"extended": True}, random_state=1)
ica.fit(raw_tmp)
ica.plot_components(inst=raw_tmp, picks=range(22))
ica.exclude = [1]
raw_corrected = raw.copy()
ica.apply(raw_corrected)
raw.plot(n_channels=25, start=53, duration=5, title="Before")
raw_corrected.plot(n_channels=25, start=53, duration=5, title="After")