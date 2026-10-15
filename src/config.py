"""Rutas, hiperparametros y constantes del proyecto."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
SAMPLES_DIR = ARTIFACTS_DIR / "samples"

# Dataset
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/"
    "smsspamcollection.zip"
)
ZIP_NAME = "smsspamcollection.zip"
RAW_NAME = "SMSSpamCollection"
RAW_PATH = DATA_DIR / RAW_NAME

# Artefactos
MODEL_PATH = ARTIFACTS_DIR / "spam_lstm.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.json"
HISTORY_PATH = ARTIFACTS_DIR / "history.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Semilla global para reproducibilidad
SEED = 42

# Etiquetas: ham -> 0, spam -> 1
CLASS_NAMES = ["ham", "spam"]
CLASS_NAMES_ES = ["Legitimo (ham)", "Spam"]
NUM_CLASSES = 2

# Division de datos (estratificada)
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Preprocesamiento de texto
MAX_VOCAB = 10_000
MAX_LEN = 160
OOV_TOKEN = "<OOV>"

# Modelo
EMBEDDING_DIM = 128
LSTM_UNITS = 128
DENSE_UNITS = 64
SPATIAL_DROPOUT = 0.2
DROPOUT = 0.5

# Entrenamiento
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 4
LR_PATIENCE = 2

# Numero minimo de predicciones de ejemplo que exige la practica
NUM_EXAMPLES = 10