import tensorflow as tf
from keras.layers import Input, Reshape, Conv2D, Flatten, SimpleRNN, LSTM, Dense, BatchNormalization, Dropout, Permute, \
    Conv1D, MaxPooling1D, Activation


def SimpleRNN_tf():
    input_shape = (1176,1)

    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Reshape((14, 84)),  # sequences of length 14 with 84 features

        Conv1D(32, kernel_size=3, activation='relu'),
        Conv1D(64, kernel_size=3, activation='relu'),
        Flatten(),

        # Reshape for RNN input
        Reshape((10, 64)),  # Adjusted shape based on Conv1D output

        SimpleRNN(32, return_sequences=True),
        LSTM(32),
        Dense(8, activation='sigmoid')
    ])

    return model


def ClassicCNN_1D_model():
    input_shape = (1176, 1)

    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Reshape((1176, 1)),

        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        Conv1D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        Conv1D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(256, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        Conv1D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv1D(512, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8, activation='sigmoid')  # 8 units for binary multi-label output
    ])

    return model

    return model

model = ClassicCNN_1D_model()

def ComplexRNN_tf():
    input_shape = (1176, 1)  # Adjusting the shape for 1D convolution

    model = tf.keras.Sequential([
        Input(shape=input_shape),

        # 1D Convolution layers
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.2),

        # RNN layers
        SimpleRNN(128, return_sequences=True, activation='relu', dropout=0.3, recurrent_dropout=0.3),
        LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        LSTM(64, dropout=0.3, recurrent_dropout=0.3),
        Flatten(),  # Flatten the output for dense layers

        # Dense layers
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(8, activation='sigmoid')  # Output layer
    ])

    return model

