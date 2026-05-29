from typing import Tuple

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src import config


def build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential(
        [
            LSTM(config.LSTM_UNITS, input_shape=input_shape),
            Dropout(config.DROPOUT_RATE),
            Dense(config.PREDICTION_DIM),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss="mse")
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Sequential:
    model = build_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    return model
