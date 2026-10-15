"""Definicion del modelo LSTM para clasificacion de spam."""

from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_model(vocab_size: int, max_len: int = config.MAX_LEN) -> keras.Sequential:
    """Embedding -> SpatialDropout -> LSTM -> Dense -> Dense(sigmoid)."""
    modelo = keras.Sequential(
        [
            keras.Input(shape=(max_len,)),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=config.EMBEDDING_DIM,
                name="embedding",
            ),
            layers.SpatialDropout1D(config.SPATIAL_DROPOUT, name="spatial_dropout"),
            layers.LSTM(
                config.LSTM_UNITS,
                dropout=config.DROPOUT,
                recurrent_dropout=0.0,
                name="lstm",
            ),
            layers.Dense(config.DENSE_UNITS, activation="relu", name="densa"),
            layers.Dropout(config.DROPOUT, name="dropout"),
            layers.Dense(1, activation="sigmoid", name="salida"),
        ],
        name="clasificador_spam_lstm",
    )
    return modelo


def compile_model(modelo: keras.Sequential) -> keras.Sequential:
    """Compila con binary crossentropy.

    No se usan las metricas de Keras Precision/Recall porque en TensorFlow 2.22
    fallan con este problema; toda la evaluacion se hace con scikit-learn.
    """
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[],
    )
    return modelo


def build_callbacks(checkpoint_path) -> list:
    """EarlyStopping y reduccion de LR, ambos sobre val_loss."""
    return [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=config.LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
    ]