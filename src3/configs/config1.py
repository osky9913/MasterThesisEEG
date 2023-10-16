from src2.dataset import DASPSDataset2

DEVICE = 'cpu'
EEG_PATH = "../datasets/Dasps.mat/"
LABEL_PATH = "../datasets/participant_rating_public.xlsx"
TRAIN_RANGE = range(1, 15)
TEST_RANGE = range(15, 20)
VALID_RANGE = range(20, 24)
FLAG_CATEGORICAL_BIN = True,
FLAG_CATEGORICAL_ENC = True,

FLAG_CLEAN = True
FILTER_FLAG = True
WAVELET_FLAG = False
ICA_FLAG = True
ICA_METHOD = 'infomax'
L_FREQ = 1
H_FREQ = None
MIN_ROCKET_FLAG = False
NUMBER_FEATURE_ROCKET = 100
NUMBER_SPS_ROBUST = 128
NEW_FEAT_FLAG = True


train_dataset = DASPSDataset2(
    EEG_PATH, LABEL_PATH,
    TRAIN_RANGE,
    flag_categorical_bin=FLAG_CATEGORICAL_BIN,
    flag_categorical_enc=FLAG_CATEGORICAL_ENC,
    flag_psd=False,
    flag_fft=False,
    flag_clean=FLAG_CLEAN,
    filter_flag=FILTER_FLAG,
    flag_wavelet=WAVELET_FLAG,
    flag_ica=ICA_FLAG,
    ica_method=ICA_METHOD,
    l_freq=L_FREQ,
    h_freq=H_FREQ,
    flag_min_rocket=MIN_ROCKET_FLAG,
    number_features_rocket=NUMBER_FEATURE_ROCKET,
    number_sps_robust=NUMBER_SPS_ROBUST,
    flag_new_feat=NEW_FEAT_FLAG,
    device=DEVICE
)
