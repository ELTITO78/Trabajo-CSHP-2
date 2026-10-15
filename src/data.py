"""Carga, limpieza y preprocesamiento del corpus SMS Spam Collection."""

import json
import re
import zipfile
from urllib.request import urlopen

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src import config

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MONEY_RE = re.compile(r"[$£€]\s?\d[\d.,]*")
_NUMBER_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^a-z\s]")
_SPACES_RE = re.compile(r"\s+")


def download_dataset(force: bool = False) -> config.Path:
    """Descarga y extrae el corpus si todavia no esta en disco."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_PATH.exists() and not force:
        return config.RAW_PATH

    zip_path = config.DATA_DIR / config.ZIP_NAME
    if force or not zip_path.exists():
        print(f"Descargando dataset desde {config.DATA_URL}")
        with urlopen(config.DATA_URL, timeout=120) as respuesta:
            zip_path.write_bytes(respuesta.read())

    with zipfile.ZipFile(zip_path) as archivo:
        archivo.extract(config.RAW_NAME, config.DATA_DIR)

    if not config.RAW_PATH.exists():
        raise FileNotFoundError("No se encontro SMSSpamCollection en el zip")

    print(f"Dataset listo en {config.RAW_PATH}")
    return config.RAW_PATH


def load_raw() -> tuple[list[str], list[str]]:
    """Lee el fichero TSV y devuelve (mensajes, etiquetas crudas)."""
    lineas = config.RAW_PATH.read_text(encoding="utf-8").splitlines()
    mensajes, etiquetas = [], []

    for linea in lineas:
        if "\t" not in linea:
            continue
        etiqueta, mensaje = linea.split("\t", 1)
        mensajes.append(mensaje)
        etiquetas.append(etiqueta.strip().lower())

    return mensajes, etiquetas


def clean_text(mensajes: list[str]) -> list[str]:
    """Normaliza el texto: minusculas, sin URLs ni numeros, solo letras."""
    limpios = []
    for texto in mensajes:
        texto = texto.lower()
        texto = _URL_RE.sub(" <url> ", texto)
        texto = _EMAIL_RE.sub(" <email> ", texto)
        texto = _MONEY_RE.sub(" <dinero> ", texto)
        texto = _NUMBER_RE.sub(" <num> ", texto)
        texto = _PUNCT_RE.sub(" ", texto)
        texto = _SPACES_RE.sub(" ", texto).strip()
        limpios.append(texto)
    return limpios


def build_tokenizer(mensajes: list[str]) -> keras.Tokenizer:
    """Ajusta un Tokenizer sobre el vocabulario del conjunto de entrenamiento."""
    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=config.MAX_VOCAB,
        filters="",
        oov_token=config.OOV_TOKEN,
    )
    tokenizer.fit_on_texts(mensajes)
    return tokenizer


def encode(mensajes: list[str], tokenizer: keras.Tokenizer) -> np.ndarray:
    """Convierte los mensajes en matrices de enteros con padding a la izquierda."""
    secuencias = tokenizer.texts_to_sequences(mensajes)
    return keras.preprocessing.sequence.pad_sequences(
        secuencias,
        maxlen=config.MAX_LEN,
        padding="pre",
        truncating="post",
    )


def load_tokenizer(path=config.TOKENIZER_PATH) -> keras.Tokenizer:
    """Recupera el Tokenizer guardado durante el entrenamiento."""
    if not path.exists():
        raise FileNotFoundError(
            "No se encuentra el tokenizer. Ejecuta primero `python train.py`."
        )
    guardado = json.loads(path.read_text(encoding="utf-8"))
    cfg = guardado.get("config", guardado)

    # Keras guarda estos mapas como cadenas JSON dentro del JSON.
    def _mapa(clave):
        valor = cfg.get(clave) or {}
        if isinstance(valor, str):
            valor = json.loads(valor)
        return valor

    tokenizer = keras.preprocessing.text.Tokenizer(
        num_words=cfg.get("num_words") or config.MAX_VOCAB,
        filters=cfg.get("filters", ""),
        oov_token=cfg.get("oov_token") or config.OOV_TOKEN,
    )
    tokenizer.word_index = _mapa("word_index")
    tokenizer.word_counts = _mapa("word_counts")
    tokenizer.index_docs = _mapa("index_docs")
    tokenizer.document_count = cfg.get("document_count", 0)
    return tokenizer


def predict_text(
    modelo, tokenizer, texto: str, max_len: int = config.MAX_LEN
) -> float:
    """Devuelve la probabilidad de que un mensaje sea spam."""
    limpio = clean_text([texto])[0]
    x = encode([limpio], tokenizer)[:, :max_len]
    return float(modelo.predict(x, verbose=0)[0][0])


def split_indices(
    n: int, y: np.ndarray, seed: int = config.SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve los indices de entrenamiento, validacion y prueba."""
    indices = np.arange(n)
    idx_ent, idx_temp = train_test_split(
        indices, test_size=config.TEST_SIZE + config.VAL_SIZE, stratify=y, random_state=seed
    )
    # La proporcion se aplica sobre lo que queda, no sobre el total.
    prop_val = config.VAL_SIZE / (config.VAL_SIZE + config.TEST_SIZE)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=1 - prop_val, stratify=y[idx_temp], random_state=seed
    )
    return idx_ent, idx_val, idx_test


def prepare(force_download: bool = False) -> dict:
    """Pipeline completo: descarga -> limpieza -> tokenizado -> division."""
    download_dataset(force=force_download)

    mensajes, etiquetas_crudas = load_raw()
    y = np.array([1 if e == "spam" else 0 for e in etiquetas_crudas], dtype="float32")
    mensajes_limpios = clean_text(mensajes)

    idx_ent, idx_val, idx_test = split_indices(len(mensajes), y)

    # El tokenizer se ajusta solo con entrenamiento para no filtrar informacion.
    tokenizer = build_tokenizer([mensajes_limpios[i] for i in idx_ent])
    x = encode(mensajes_limpios, tokenizer)

    return {
        "x_train": x[idx_ent],
        "x_val": x[idx_val],
        "x_test": x[idx_test],
        "y_train": y[idx_ent],
        "y_val": y[idx_val],
        "y_test": y[idx_test],
        "idx_train": idx_ent,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "tokenizer": tokenizer,
        "mensajes": mensajes,
        "mensajes_limpios": mensajes_limpios,
        "etiquetas": y,
    }