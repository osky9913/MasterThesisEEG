import json
import os
import logging
import pickle
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from lightning_fabric.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime
import argparse

from torch.utils.data import DataLoader

from dataset import DASPSDataset2
import pytorch_lightning as pl

from utils import convert_pytorch_dataset_to_tf_dataset
from torch_models import SimpleLinear, SimpleCNN, SimpleRNN
from tensor_models import SimpleRNN_tf, ComplexRNN_tf, ClassicCNN_1D_model


def load_config(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config


def get_model_by_name(name, learning_rate):
    if name == "SimpleLinear":
        return SimpleLinear(learning_rate)
    elif name == "SimpleCNN":
        return SimpleCNN(learning_rate)
    elif name == "SimpleRNN":
        return SimpleRNN(learning_rate)
    else:
        raise ValueError(f"Model {name} not recognized!")


def get_model_by_name_tf(name, ):
    if name == "SimpleRNN":
        return SimpleRNN_tf()

    if name == "ComplexRNN":
        return ComplexRNN_tf()

    if name == "ClassicCNN_1D_model":
        return ClassicCNN_1D_model()
    else:
        raise ValueError(f"Model {name} not recognized!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config1.json", help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    today = datetime.now()
    datestring = today.strftime("%Y_%m_%d_%H_%M_%S")

    path = os.path.join(config["logs_dir"], config["run_name"], datestring)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    filename_log = os.path.join(path, config["logs_name"])
    filename_log_result = os.path.join(path, config["logs_result"])
    filename_model_weights = os.path.join(path, config["model_save_name"])
    filename_plot_name = os.path.join(path,config["plot_name"])

    logging.basicConfig(filename=filename_log, filemode='w+', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info("Training starting")
    logging.info("Showing configuration")
    for key in config.keys():
        logging.info(f'{key}: {config[key]}')

    logging.info("----------------------------------------------------------------------------------------------------")

    logging.info("Info about the training dataset")
    logging.info("----------------------------------------------------------------------------------------------------")

    train_dataset = DASPSDataset2(
        config["EEG_PATH"],
        config["LABEL_PATH"],
        range(config["TRAIN_RANGE"][0], config["TRAIN_RANGE"][1]),
        flag_categorical_bin=config["FLAG_CATEGORICAL_BIN"],
        flag_categorical_enc=config["FLAG_CATEGORICAL_ENC"],
        flag_categorical_bin_bce=config["FLAG_CATEGORICAL_BIN_BCE"],
        flag_psd=False,
        flag_fft=False,
        flag_clean=config["FLAG_CLEAN"],
        flag_filter=config["FLAG_FILTER"],
        flag_wavelet=config["FLAG_WAVELET"],
        flag_ica=config["FLAG_ICA"],
        ica_method=config["ICA_METHOD"],
        l_freq=config["L_FREQ"],
        h_freq=config["H_FREQ"],
        flag_min_rocket=config["FLAG_MIN_ROCKET"],
        number_features_rocket=config["NUMBER_FEATURE_ROCKET"],
        number_sps_robust=config["NUMBER_SPS_ROBUST"],
        flag_new_feat=config["FLAG_NEW_FEAT"],
        eeg_splitter_sec=config["eeg_splitter_sec"],
        device=config["DEVICE"]
    )

    logging.info("All Y values for ham1, ham2 {}".format(train_dataset.all_labels))
    logging.info("len of the dataset {}".format(len(train_dataset)))
    feature, label = train_dataset[0]
    logging.info(feature.shape)
    logging.info(label.shape)
    batch_size = config.get("batch_size", 32)
    logging.info("Batch size: {}".format(batch_size))
    # pprint.pprint(dataset.custom_get_item(0))

    test_dataset = DASPSDataset2(
        config["EEG_PATH"],
        config["LABEL_PATH"],
        range(config["TEST_RANGE"][0], config["TEST_RANGE"][1]),
        flag_categorical_bin=config["FLAG_CATEGORICAL_BIN"],
        flag_categorical_enc=config["FLAG_CATEGORICAL_ENC"],
        flag_categorical_bin_bce=config["FLAG_CATEGORICAL_BIN_BCE"],
        flag_psd=False,
        flag_fft=False,
        flag_clean=config["FLAG_CLEAN"],
        flag_filter=config["FLAG_FILTER"],
        flag_wavelet=config["FLAG_WAVELET"],
        flag_ica=config["FLAG_ICA"],
        ica_method=config["ICA_METHOD"],
        l_freq=config["L_FREQ"],
        h_freq=config["H_FREQ"],
        flag_min_rocket=config["FLAG_MIN_ROCKET"],
        number_features_rocket=config["NUMBER_FEATURE_ROCKET"],
        number_sps_robust=config["NUMBER_SPS_ROBUST"],
        flag_new_feat=config["FLAG_NEW_FEAT"],
        eeg_splitter_sec=config["eeg_splitter_sec"],
        device=config["DEVICE"]
    )

    valid_dataset = DASPSDataset2(
        config["EEG_PATH"],
        config["LABEL_PATH"],
        range(config["VALID_RANGE"][0], config["VALID_RANGE"][1]),
        flag_categorical_bin=config["FLAG_CATEGORICAL_BIN"],
        flag_categorical_enc=config["FLAG_CATEGORICAL_ENC"],
        flag_categorical_bin_bce=config["FLAG_CATEGORICAL_BIN_BCE"],
        flag_psd=False,
        flag_fft=False,
        flag_clean=config["FLAG_CLEAN"],
        flag_filter=config["FLAG_FILTER"],
        flag_wavelet=config["FLAG_WAVELET"],
        flag_ica=config["FLAG_ICA"],
        ica_method=config["ICA_METHOD"],
        l_freq=config["L_FREQ"],
        h_freq=config["H_FREQ"],
        flag_min_rocket=config["FLAG_MIN_ROCKET"],
        number_features_rocket=config["NUMBER_FEATURE_ROCKET"],
        number_sps_robust=config["NUMBER_SPS_ROBUST"],
        flag_new_feat=config["FLAG_NEW_FEAT"],
        eeg_splitter_sec=config["eeg_splitter_sec"],
        device=config["DEVICE"]
    )
    learning_rate = config.get("learning_rate", 0.00001)
    max_epochs = config.get("max_epochs", 100),

    logging.info("learning_rate {}".format(learning_rate))
    logging.info("max_epochs {}".format(max_epochs))
    if config["tensorflow"] == False:

        logging.info("Using pytorch lighting backend ")
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

        model_name = config.get("model", "SimpleLinear")
        model = get_model_by_name(model_name, learning_rate)

        # Checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            mode='min',
        )
        logger = TensorBoardLogger("tb_logs", name="my_model")

        # Trainer setup and training
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback],
            accelerator='mps',
            log_every_n_steps=1
        )
        trainer.fit(model, train_dataloader, valid_dataloader, )

        # Evaluation on test set
        trainer.test(model, test_dataloader)
    else:

        logging.info("Using tensorflow backend ")
        train_x, train_y = convert_pytorch_dataset_to_tf_dataset(train_dataset)
        test_x, test_y = convert_pytorch_dataset_to_tf_dataset(test_dataset)
        valid_x, valid_y = convert_pytorch_dataset_to_tf_dataset(valid_dataset)

        logging.info("Test of dataset if inputs are of the same shape")

        first_x = train_x[0]
        for x in train_x:
            assert first_x.shape == x.shape

        logging.info("Tested")
        print(first_x)
        print(first_x.shape)

        max_epochs = max_epochs[0]
        # Define the RNN model

        logging.info("Loading the model")
        model_name = config.get("model", "SimpleLinear")
        model = get_model_by_name_tf(name=model_name)
        logging.info("Loaded")

        # Compile the model with categorical cross-entropy loss
        logging.info("Compiling")

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Print a summary of the model architecture
        model.summary()

        with tf.device('/device:GPU:0'):
            history = model.fit(x=train_x, y=train_y, epochs=max_epochs, batch_size=batch_size,validation_split=0.3 , shuffle=True)

            #validation_data = (test_x, test_y)

        logging.info("History: {}".format(history.history))
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        logging.info("accuracy: {}".format(accuracy))
        logging.info("val_accuracy: {}".format(val_accuracy))
        logging.info("loss: {}".format(loss))
        logging.info("val_loss: {}".format(val_loss))



        logging.info("Saving values to {}".format(filename_log_result))
        file = open(filename_log_result, 'wb')
        pickle.dump(history.history, file)
        file.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('model accuracy')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='upper left')

        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'test'], loc='upper left')

        plt.tight_layout()
        plt.show()
        fig.savefig(filename_plot_name)
